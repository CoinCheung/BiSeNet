
export CUDA_VISIBLE_DEVICES=2,3
PORT=52334
NGPUS=2
cfg_file=configs/bisenetv1_city.py
# cfg_file=configs/bisenetv2_city.py
# cfg_file=configs/bisenetv2_coco.py

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/evaluate.py --config $cfg_file --port $PORT --weight-path res/model_final.pth
