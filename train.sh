TRAINER_NAME=ddppo-e3b  # ddppo|ddppo-icm|ddppo-e3b

rm -rf data/hm3d/$TRAINER_NAME/train

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=1 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d.yaml \
    habitat_baselines.trainer_name=$TRAINER_NAME \
    habitat.seed=100 \
    habitat_baselines.tensorboard_dir="data/hm3d/$TRAINER_NAME/train/tb" \
    habitat_baselines.eval_ckpt_path_dir="data/hm3d/$TRAINER_NAME/train/new_checkpoints" \
    habitat_baselines.checkpoint_folder="data/hm3d/$TRAINER_NAME/train/new_checkpoints" \
    habitat_baselines.log_file="data/hm3d/$TRAINER_NAME/train/train.log" \
    habitat_baselines.log_interval=10 \
    habitat_baselines.num_checkpoints=100 \
    habitat_baselines.num_environments=40 \
    habitat_baselines.total_num_steps=2e7
