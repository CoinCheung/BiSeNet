

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3
PORT=12333
NGPUS=2
cfg=configs/bisenetv1_city.py
# cfg=configs/bisenetv2_city.py
# cfg=configs/bisenetv1_coco.py
# cfg=configs/bisenetv2_coco.py

torchrun --nproc_per_node=$NGPUS --master_port $PORT tools/train_amp.py --config $cfg
# python -m torch.distributed.launch --use_env --nproc_per_node=$NGPUS --master_port $PORT tools/train_amp.py --config $cfg

