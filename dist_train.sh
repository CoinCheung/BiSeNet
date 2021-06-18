
export CUDA_VISIBLE_DEVICES=2,3
PORT=52330
NGPUS=2
cfg_file=configs/bisenetv1_city.py

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train.py --config $cfg_file --port $PORT
