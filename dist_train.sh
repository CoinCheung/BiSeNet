
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=52339
NGPUS=8
# cfg_file=configs/bisenetv1_city.py
# cfg_file=configs/bisenetv2_city.py
cfg_file=configs/bisenetv2_coco.py

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/evaluate.py --config $cfg_file --port $PORT --weight-path res/model_final.pth
