TRAINER_NAME=ddppo-tdd  # ddppo|ddppo-icm|ddppo-e3b|ddppo-rnd|ddppo-noveld|ddppo-tdd
EVAL_DATA_SPLIT=val
SEED=100
ROOT_DIR="data/hm3d/$TRAINER_NAME/eval-$EVAL_DATA_SPLIT/seed=$SEED"
CKPT_DIR="data/hm3d/$TRAINER_NAME/train/seed=100/checkpoints/ckpt.82.pth"

MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=0 python -u -m habitat_baselines.run \
    --config-name=pointnav/ddppo_pointnav_hm3d_eval.yaml \
    habitat_baselines.trainer_name=$TRAINER_NAME \
    habitat.seed=$SEED \
    habitat.dataset.split=$EVAL_DATA_SPLIT \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval_ckpt_path_dir=$CKPT_DIR \
    habitat_baselines.checkpoint_folder=$ROOT_DIR \
    habitat_baselines.video_dir="$ROOT_DIR/videos" \
    habitat_baselines.tensorboard_dir="$ROOT_DIR/tb" \
    habitat_baselines.log_file="$ROOT_DIR/eval.log" \
    habitat_baselines.test_episode_count=5 \
    habitat_baselines.num_environments=1
