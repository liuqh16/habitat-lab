#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from habitat.utils import profiling_wrapper
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ddppo.policy import (
    PointNavResNetNet,
    InverseDynamicsNet,
    ForwardDynamicsNet,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer


@baseline_registry.register_trainer(name="ddppo-icm")
class ICMTrainer(PPOTrainer):
    r"""Trainer class for PPO algorithm with ICM.
    Paper: https://arxiv.org/abs/1705.05363.
    """
    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        self._create_obs_transforms()

        self.encoder = PointNavResNetNet(observation_space=self._env_spec.observation_space,
                                         action_space=None,
                                         hidden_size=self.config.habitat_baselines.rl.ppo.hidden_size,
                                         num_recurrent_layers=0,
                                         rnn_type='none',
                                         backbone=self.config.habitat_baselines.rl.ddppo.backbone,
                                         resnet_baseplanes=32,
                                         normalize_visual_inputs="rgb" in self._env_spec.observation_space.spaces,
                                         fuse_keys=None,
                                         force_blind_policy=self.config.habitat_baselines.force_blind_policy)
        self.encoder.to(self.device)

        self.inverse_dynamics = InverseDynamicsNet(num_actions=self._env_spec.action_space.n,
                                                   hidden_size=self.config.habitat_baselines.rl.ppo.hidden_size)
        self.inverse_dynamics.to(self.device)

        self.forward_dynamics = ForwardDynamicsNet(num_actions=self._env_spec.action_space.n,
                                                   hidden_size=self.config.habitat_baselines.rl.ppo.hidden_size)
        self.forward_dynamics.to(self.device)

        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                                  lr=self.config.habitat_baselines.rl.explore.model_learning_rate)
        self.inverse_dynamics_optimizer = torch.optim.Adam(self.inverse_dynamics.parameters(),
                                                           lr=self.config.habitat_baselines.rl.explore.model_learning_rate)
        self.forward_dynamics_optimizer = torch.optim.Adam(self.forward_dynamics.parameters(),
                                                           lr=self.config.habitat_baselines.rl.explore.model_learning_rate)

        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

    def _init_train(self, resume_state=None):
        super()._init_train(resume_state)
        # ICM objects
        self.running_episode_stats["bonus_reward"] = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_bonus_reward = torch.zeros_like(self.current_episode_reward)
        self.reward_bonus_buffer = torch.zeros_like(self.current_episode_reward)

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # Sample actions
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            next_step_batch = self._agent.rollouts.get_next_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            # Obtain lenghts
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

            # Compute intrinsic reward bonus
            phi = self.encoder(step_batch["observations"], None, None, None, None)[0]
            pred_next_phi = self.forward_dynamics(phi, step_batch["actions"])
            next_phi = self.encoder(next_step_batch["observations"], None, None, None, None)[0]
            bonus = F.mse_loss(pred_next_phi, next_phi, reduction="none").mean(1)
            self.reward_bonus_buffer.copy_(bonus.unsqueeze(1))

        profiling_wrapper.range_pop()  # compute actions

        with g_timer.avg_time("trainer.obs_insert"):
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                if is_continuous_action_space(self._env_spec.action_space):
                    # Clipping actions to the specified limits
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    act = act.item()
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            if self.config.habitat_baselines.reward_free:
                rewards.zero_()

            bonus_rewards = self.reward_bonus_buffer.squeeze().clone().detach().unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            self.current_episode_bonus_reward[env_slice] += bonus_rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            current_ep_bonus_reward = self.current_episode_bonus_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
            self.running_episode_stats["bonus_reward"][env_slice] += current_ep_bonus_reward.where(done_masks, current_ep_bonus_reward.new_zeros(()))  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )
            self.current_episode_bonus_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

            # Add intrinsic and extrinsic rewards together
            rewards = rewards + self.config.habitat_baselines.rl.explore.int_rew_coef * bonus_rewards

        if self._is_static_encoder:
            with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                batch[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ] = self._encoder(batch)

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        losses = self._agent.updater.update(self._agent.rollouts)

        # ICM stuff
        inv_dynamics_loss, fwd_dynamics_loss = 0, 0
        for i in range(self.config.habitat_baselines.rl.explore.model_n_epochs):

            # Compute IDM loss
            self.encoder_optimizer.zero_grad()
            self.inverse_dynamics_optimizer.zero_grad()
            self.forward_dynamics_optimizer.zero_grad()
            obs, action, next_obs = self._agent.rollouts.sample_triplet()
            phi = self.encoder(obs, None, None, None, None)[0]
            next_phi = self.encoder(next_obs, None, None, None, None)[0]
            action_pred = F.log_softmax(self.inverse_dynamics(phi, next_phi), dim=-1)
            idm_loss = F.nll_loss(action_pred, action.squeeze())

            # Compute FDM loss
            pred_next_phi = self.forward_dynamics(phi, action)
            fdm_loss = F.mse_loss(pred_next_phi, next_phi)

            loss = idm_loss * (1 - self.config.habitat_baselines.rl.explore.icm.forward_loss_coef) + \
                   fdm_loss * self.config.habitat_baselines.rl.explore.icm.forward_loss_coef

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            self.encoder_optimizer.step()
            self.inverse_dynamics_optimizer.step()
            self.forward_dynamics_optimizer.step()
            inv_dynamics_loss += idm_loss.item()
            fwd_dynamics_loss += fdm_loss.item()

        if self.config.habitat_baselines.rl.explore.model_n_epochs > 0:
            inv_dynamics_loss /= self.config.habitat_baselines.rl.explore.model_n_epochs
            fwd_dynamics_loss /= self.config.habitat_baselines.rl.explore.model_n_epochs

        losses["inv_dynamics_loss"] = inv_dynamics_loss
        losses["fwd_dynamics_loss"] = fwd_dynamics_loss

        self._agent.rollouts.after_update()
        self._agent.after_update()

        return losses
