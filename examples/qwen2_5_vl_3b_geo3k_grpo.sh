set -x
NCCL_DEBUG=WARN 
MODEL_PATH=/home/qbw/research/RL/EasyR1/cache/ckpt/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/qbw/research/RL/EasyR1/cache/data/geometry3k@train \
    data.val_files=/home/qbw/research/RL/EasyR1/cache/data/geometry3k@test \
    data.max_pixels=1048576 \
    data.min_pixels=262144 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.rollout.tensor_parallel_size=2 \
    trainer.experiment_name=qwen2_5_vl_3b_geo3k_grpo \
    trainer.n_gpus_per_node=4
