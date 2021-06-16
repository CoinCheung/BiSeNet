
export CUDA_VISIBLE_DEVICES=6,7
PORT=52332
NGPUS=2

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --model bisenetv2 --port $PORT
