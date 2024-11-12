TRAINER_NAME=ddppo-tdd  # ddppo|ddppo-icm|ddppo-e3b|ddppo-rnd|ddppo-noveld|ddppo-tdd
SEED=100
ROOT_DIR="data/hm3d/$TRAINER_NAME/train/seed=$SEED"

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d.yaml \
    habitat_baselines.trainer_name=$TRAINER_NAME \
    habitat.seed=$SEED \
    habitat_baselines.tensorboard_dir="$ROOT_DIR/tb" \
    habitat_baselines.eval_ckpt_path_dir="$ROOT_DIR/checkpoints" \
    habitat_baselines.checkpoint_folder="$ROOT_DIR/checkpoints" \
    habitat_baselines.log_file="$ROOT_DIR/train.log" \
    habitat_baselines.log_interval=10 \
    habitat_baselines.num_checkpoints=100 \
    habitat_baselines.num_environments=40 \
    habitat_baselines.total_num_steps=2e7 \
    habitat_baselines.rl.explore.int_rew_coef=0.1 \
    habitat_baselines.rl.explore.model_learning_rate=2.5e-4 \
    habitat_baselines.rl.explore.model_n_epochs=3 \
    habitat_baselines.rl.explore.tdd.aggregate_fn=min \
    habitat_baselines.rl.explore.tdd.energy_fn=mrn_pot \
    habitat_baselines.rl.explore.tdd.loss_fn=infonce_backward \
    habitat_baselines.rl.explore.tdd.discount=0.99 \
    habitat_baselines.rl.explore.tdd.temperature=1.0 \
    habitat_baselines.rl.explore.tdd.logsumexp_coef=0.1 \
    habitat_baselines.rl.explore.tdd.knn_k=10
