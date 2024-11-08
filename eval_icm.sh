TRAINER_NAME=ddppo-icm

rm -rf data/hm3d/$TRAINER_NAME/eval

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d.yaml \
    habitat_baselines.trainer_name=$TRAINER_NAME \
    habitat.seed=100 \
    habitat_baselines.evaluate=True \
    habitat_baselines.video_dir="data/hm3d/$TRAINER_NAME/eval/videos" \
    habitat_baselines.tensorboard_dir="data/hm3d/$TRAINER_NAME/eval/tb" \
    habitat_baselines.eval_ckpt_path_dir="data/hm3d/$TRAINER_NAME/train/new_checkpoints/ckpt.0.pth" \
    habitat_baselines.log_file="data/hm3d/$TRAINER_NAME/eval/eval.log" \
    habitat_baselines.num_environments=1
