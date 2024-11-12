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
    ProjectionNet,
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


def mrn_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Metric Residual Network (MRN) architecture (https://arxiv.org/pdf/2208.08133)
    x_prefix, x_suffix = x.chunk(2, dim=-1)
    y_prefix, y_suffix = y.chunk(2, dim=-1)
    max_component = (x_prefix - y_prefix).relu().max(dim=-1).values
    l2_component = ((x_suffix - y_suffix).pow(2).sum(dim=-1) + eps).sqrt()
    return max_component + l2_component


@baseline_registry.register_trainer(name="ddppo-tdd")
class TDDTrainer(PPOTrainer):
    r"""Trainer class for PPO algorithm with TDD.
    """
    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        self._create_obs_transforms()

        self.feat_extractor = PointNavResNetNet(observation_space=self._env_spec.observation_space,
                                                action_space=None,
                                                hidden_size=self.config.habitat_baselines.rl.ppo.hidden_size,
                                                num_recurrent_layers=0,
                                                rnn_type='none',
                                                backbone=self.config.habitat_baselines.rl.ddppo.backbone,
                                                resnet_baseplanes=32,
                                                normalize_visual_inputs="rgb" in self._env_spec.observation_space.spaces,
                                                fuse_keys=None,
                                                force_blind_policy=self.config.habitat_baselines.force_blind_policy)
        self.feat_extractor.to(self.device)

        self.encoder = ProjectionNet(input_size=self.config.habitat_baselines.rl.ppo.hidden_size,
                                     output_size=self.config.habitat_baselines.rl.ppo.hidden_size)
        self.encoder.to(self.device)

        self.potential = ProjectionNet(input_size=self.config.habitat_baselines.rl.ppo.hidden_size,
                                       output_size=1)
        self.potential.to(self.device)

        self.model_optimizer = torch.optim.Adam([
                {"params": self.feat_extractor.parameters()},
                {"params": self.encoder.parameters()},
                {"params": self.potential.parameters()},
            ], lr=self.config.habitat_baselines.rl.explore.model_learning_rate)

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
        # TDD objects
        self.running_episode_stats["bonus_reward"] = torch.zeros(self.envs.num_envs, 1)
        self.current_episode_bonus_reward = torch.zeros_like(self.current_episode_reward)
        self.reward_bonus_buffer = torch.zeros_like(self.current_episode_reward)
        self.obs_embedding_buffer = [None for _ in range(self.envs.num_envs)]

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
            curr_obs_embs: torch.Tensor = self.feat_extractor(step_batch["observations"], None, None, None, None)[0]
            next_obs_embs: torch.Tensor = self.feat_extractor(next_step_batch["observations"], None, None, None, None)[0]

            for index_env, curr_obs_emb, next_obs_emb in zip(
                range(env_slice.start, env_slice.stop),
                curr_obs_embs.unbind(0),
                next_obs_embs.unbind(0),
            ):
                # Update historical observation embeddings
                obs_embs = self.obs_embedding_buffer[index_env]
                if obs_embs is None:
                    obs_embs = torch.stack([curr_obs_emb, next_obs_emb], dim=0)
                else:
                    obs_embs = torch.cat([obs_embs, next_obs_emb.unsqueeze(0)], dim=0)
                self.obs_embedding_buffer[index_env] = obs_embs

                # Compute dists
                phi_x: torch.Tensor = self.encoder(obs_embs[:-1])
                phi_y: torch.Tensor = self.encoder(obs_embs[-1:])
                if self.config.habitat_baselines.rl.explore.tdd.energy_fn == "l2":
                    dists = torch.sqrt((phi_x[:, None, :] - phi_y[None, :, :]).pow(2).sum(dim=-1) + 1e-8)
                elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "cos":
                    phi_x_norm = F.normalize(phi_x, p=2, dim=-1) / self.config.habitat_baselines.rl.explore.tdd.temperature
                    phi_y_norm = F.normalize(phi_y, p=2, dim=-1)
                    dists = - torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
                elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "dot":
                    dists = - torch.einsum("ik,jk->ij", phi_x, phi_y)
                elif "mrn" in self.config.habitat_baselines.rl.explore.tdd.energy_fn:
                    dists = mrn_distance(phi_x[:, None, :], phi_y[None, :, :])
                else:
                    raise NotImplementedError

                # Compute intrinsic reward
                if self.config.habitat_baselines.rl.explore.tdd.aggregate_fn == "min":
                    self.reward_bonus_buffer[index_env].copy_(dists.min(), non_blocking=True)
                elif self.config.habitat_baselines.rl.explore.tdd.aggregate_fn == "quantile10":
                    self.reward_bonus_buffer[index_env].copy_(dists.quantile(q=0.1), non_blocking=True)
                elif self.config.habitat_baselines.rl.explore.tdd.aggregate_fn == "knn":
                    knn_dists = dists if dists.size(0) <= self.config.habitat_baselines.rl.explore.tdd.knn_k \
                        else dists.topk(k=self.config.habitat_baselines.rl.explore.tdd.knn_k, largest=False).values
                    self.reward_bonus_buffer[index_env].copy_(knn_dists[-1], non_blocking=True)
                else:
                    raise NotImplementedError

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

            # Reset episodic obs-embedding buffer
            for index_env, done in zip(range(env_slice.start, env_slice.stop), dones):
                if done:
                    self.obs_embedding_buffer[index_env] = None

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

        # TDD stuff
        self._agent.rollouts.compute_future_states(discount=self.config.habitat_baselines.rl.explore.tdd.discount)
        contrastive_loss, logits_logsumexp = 0, 0
        logits_pos, logits_neg, categorical_accuracy = 0, 0, 0
        for i in range(self.config.habitat_baselines.rl.explore.model_n_epochs):

            # Compute contrastive loss
            self.model_optimizer.zero_grad()
            obs, action, next_obs = self._agent.rollouts.sample_triplet(use_future_state=True)
            curr_obs_embs = self.feat_extractor(obs, None, None, None, None)[0]
            next_obs_embs = self.feat_extractor(next_obs, None, None, None, None)[0]
            phi_x: torch.Tensor = self.encoder(curr_obs_embs)
            phi_y: torch.Tensor = self.encoder(next_obs_embs)
            c_y: torch.Tensor = self.potential(next_obs_embs)

            if self.config.habitat_baselines.rl.explore.tdd.energy_fn == "l2":
                logits = -torch.sqrt((phi_x[:, None, :] - phi_y[None, :, :]).pow(2).sum(dim=-1) + 1e-8)
            elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "cos":
                phi_x_norm = F.normalize(phi_x, p=2, dim=-1) / self.config.habitat_baselines.rl.explore.tdd.temperature
                phi_y_norm = F.normalize(phi_y, p=2, dim=-1)
                logits = torch.einsum("ik,jk->ij", phi_x_norm, phi_y_norm)
            elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "dot":
                logits = torch.einsum("ik,jk->ij", phi_x, phi_y)
            elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "mrn":
                logits = - mrn_distance(phi_x[:, None, :], phi_y[None, :, :])
            elif self.config.habitat_baselines.rl.explore.tdd.energy_fn == "mrn_pot":
                logits = c_y - mrn_distance(phi_x[:, None, :], phi_y[None, :, :])
            else:
                raise NotImplementedError

            batch_size = logits.size(0)
            I = torch.eye(batch_size, device=logits.device)
            if self.config.habitat_baselines.rl.explore.tdd.loss_fn == "infonce":
                contrast_loss = F.cross_entropy(logits, I).mean()
            elif self.config.habitat_baselines.rl.explore.tdd.loss_fn == "infonce_backward":
                contrast_loss = F.cross_entropy(logits.T, I).mean()
            elif self.config.habitat_baselines.rl.explore.tdd.loss_fn == "infonce_symmetric":
                contrast_loss = ((F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2).mean()
            elif self.config.habitat_baselines.rl.explore.tdd.loss_fn == "dpo":
                positive = torch.diag(logits)
                diffs = positive[:, None] - logits
                contrast_loss = -F.logsigmoid(diffs).mean()
            else:
                raise NotImplementedError
            logsumexp_loss = (torch.logsumexp(logits + 1e-6, dim=1).pow(2)).mean()
            loss = contrast_loss + self.config.habitat_baselines.rl.explore.tdd.logsumexp_coef * logsumexp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.feat_extractor.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.potential.parameters(), self.config.habitat_baselines.rl.ppo.max_grad_norm)
            self.model_optimizer.step()
            contrastive_loss += contrast_loss.item()
            logits_logsumexp += logsumexp_loss.item()
            logits_pos += logits.diag().mean().item()
            logits_neg += logits[~I.bool()].mean().item()
            categorical_accuracy += (logits.argmax(dim=1) == torch.arange(batch_size, device=logits.device)).float().mean().item()

        if self.config.habitat_baselines.rl.explore.model_n_epochs > 0:
            contrastive_loss /= self.config.habitat_baselines.rl.explore.model_n_epochs
            logits_logsumexp /= self.config.habitat_baselines.rl.explore.model_n_epochs
            logits_pos /= self.config.habitat_baselines.rl.explore.model_n_epochs
            logits_neg /= self.config.habitat_baselines.rl.explore.model_n_epochs
            categorical_accuracy /= self.config.habitat_baselines.rl.explore.model_n_epochs

        losses["contrastive_loss"] = contrastive_loss
        losses["logits_logsumexp"] = logits_logsumexp
        losses["logits_pos"] = logits_pos
        losses["logits_neg"] = logits_neg
        losses["categorical_accuracy"] = categorical_accuracy

        self._agent.rollouts.after_update()
        self._agent.after_update()

        return losses
