
'''
NOTE: replace torchrun with torch.distributed.launch if you use older version of pytorch. I suggest you use the same version as I do since I have not tested compatibility with older version after updating.
'''


## bisenetv1 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv1_city.py
NGPUS=2
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv2_city.py
NGPUS=2
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv1 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3
cfg_file=configs/bisenetv1_coco.py
NGPUS=4
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_coco.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv1 ade20k
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv1_ade20k.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file


## bisenetv2 ade20k
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_ade20k.py
NGPUS=8
torchrun --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file
