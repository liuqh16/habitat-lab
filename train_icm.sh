TRAINER_NAME=ddppo-icm

rm -rf data/hm3d/$TRAINER_NAME

CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d.yaml \
    habitat_baselines.tensorboard_dir="data/hm3d/$TRAINER_NAME/tb" \
    habitat_baselines.eval_ckpt_path_dir="data/hm3d/$TRAINER_NAME/new_checkpoints" \
    habitat_baselines.checkpoint_folder="data/hm3d/$TRAINER_NAME/new_checkpoints" \
    habitat_baselines.log_file="data/hm3d/$TRAINER_NAME/train.log" \
    habitat_baselines.log_interval=10 \
    habitat_baselines.num_checkpoints=100 \
    habitat_baselines.num_environments=40 \
    habitat_baselines.total_num_steps=2e7
