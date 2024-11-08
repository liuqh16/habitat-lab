TRAINER_NAME=ddppo-e3b  # ddppo|ddppo-icm|ddppo-e3b

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d_eval.yaml \
    habitat_baselines.trainer_name=$TRAINER_NAME \
    habitat.seed=100 \
    habitat_baselines.evaluate=True \
    habitat_baselines.video_dir="data/hm3d/$TRAINER_NAME/eval/videos" \
    habitat_baselines.tensorboard_dir="data/hm3d/$TRAINER_NAME/eval/tb" \
    habitat_baselines.eval_ckpt_path_dir="data/hm3d/$TRAINER_NAME/train/new_checkpoints" \
    habitat_baselines.checkpoint_folder="data/hm3d/$TRAINER_NAME/eval" \
    habitat_baselines.log_file="data/hm3d/$TRAINER_NAME/eval/eval.log" \
    habitat_baselines.num_environments=1
