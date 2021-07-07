
export CUDA_VISIBLE_DEVICES=0,1
PORT=52332
NGPUS=2
# cfg_file=configs/bisenetv1_city.py
# cfg_file=configs/bisenetv1_coco.py
# cfg_file=configs/bisenetv2_city.py
cfg_file=configs/bisenetv2_coco.py

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file --port $PORT
python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --finetune-from ./res/modelzoo/model_final_v2_city.pth --config ./configs/bisenetv2_city.py # or bisenetv1

## train, use run
# python -m torch.distributed.run --nnode=1 --rdzv_backend=c10d --rdzv_id=001 --rdzv_endpoint=127.0.0.1:$PORT --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file --port $PORT




# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train.py --config $cfg_file --port $PORT

# python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/evaluate.py --config $cfg_file --port $PORT --weight-path res/modelzoo/model_final_v2_coco.pth


