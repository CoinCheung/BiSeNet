
export CUDA_VISIBLE_DEVICES=0,1
PORT=52335
NGPUS=2

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config configs/bisenetv1_city.py --port $PORT
