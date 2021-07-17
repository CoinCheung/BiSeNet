# BiSeNetV1 & BiSeNetV2

My implementation of [BiSeNetV1](https://arxiv.org/abs/1808.00897) and [BiSeNetV2](https://arxiv.org/abs/1808.00897).


mIOUs and fps on cityscapes val set:
| none | ss | ssc | msf | mscf | fps(fp16/fp32) | link |
|------|:--:|:---:|:---:|:----:|:---:|:----:|
| bisenetv1 | 75.44 | 76.94 | 77.45 | 78.86 | 68/23 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v1_city_new.pth) |
| bisenetv2 | 74.95 | 75.58 | 76.53 | 77.08 | 59/21 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_city.pth) |


mIOUs on cocostuff val2017 set:
| none | ss | ssc | msf | mscf | link |
|------|:--:|:---:|:---:|:----:|:----:|
| bisenetv1 | 31.49 | 31.42 | 32.46 | 32.55 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v1_coco_new.pth) |
| bisenetv2 | 30.49 | 30.55 | 31.81 | 31.73 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/model_final_v2_coco.pth) |

Tips: 
1. **ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. The eval scales and crop size of multi-scales evaluation can be found in [configs](./configs/).

2. The fps is tested in different way from the paper. For more information, please see [here](./tensorrt).

3. For cocostuff dataset: The authors of the paper `bisenetv2` used the "old split" of 9k train set and 1k val set, while I used the "new split" of 118k train set and 5k val set. Thus the above results on cocostuff does not match the paper. The authors of bisenetv1 did not report their results on cocostuff, so here I simply provide a "make it work" result. Following the tradition of object detection, I used "1x"(90k) and "2x"(180k) schedule to train bisenetv1(1x) and bisenetv2(2x) respectively. Maybe you can have a better result by picking up hyper-parameters more carefully.

4. The model has a big variance, which means that the results of training for many times would vary within a relatively big margin. For example, if you train bisenetv2 for many times, you will observe that the result of **ss** evaluation of bisenetv2 varies between 73.1-75.1. 


## deploy trained models
1. tensorrt  
You can go to [tensorrt](./tensorrt) for details.

2. ncnn  
You can go to [ncnn](./ncnn) for details.


## platform
My platform is like this: 
* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.51.05
* cuda 10.2
* cudnn 7
* miniconda python 3.8.8
* pytorch 1.8.1


## get start
With a pretrained weight, you can run inference on an single image like this: 
```
$ python tools/demo.py --config configs/bisenetv2_city.py --weight-path /path/to/your/weights.pth --img-path ./example.png
```
This would run inference on the image and save the result image to `./res.jpg`.


## prepare dataset

1.cityscapes  

Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them into the `datasets/cityscapes` directory:  
```
$ mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
$ mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
$ cd datasets/cityscapes
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
```

2.cocostuff   

Download `train2017.zip`, `val2017.zip` and `stuffthingmaps_trainval2017.zip` split from official [website](https://cocodataset.org/#download). Then do as following:
```
$ unzip train2017.zip
$ unzip val2017.zip
$ mv train2017/ /path/to/BiSeNet/datasets/coco/images
$ mv val2017/ /path/to/BiSeNet/datasets/coco/images

$ unzip stuffthingmaps_trainval2017.zip
$ mv train2017/ /path/to/BiSeNet/datasets/coco/labels
$ mv val2017/ /path/to/BiSeNet/datasets/coco/labels

$ cd /path/to/BiSeNet
$ python tools/gen_coco_annos.py
```

3.custom dataset  

If you want to train on your own dataset, you should generate annotation files first with the format like this: 
```
munster_000002_000019_leftImg8bit.png,munster_000002_000019_gtFine_labelIds.png
frankfurt_000001_079206_leftImg8bit.png,frankfurt_000001_079206_gtFine_labelIds.png
...
```
Each line is a pair of training sample and ground truth image path, which are separated by a single comma `,`.   
Then you need to change the field of `im_root` and `train/val_im_anns` in the configuration files. If you found what shows in `cityscapes_cv2.py` is not clear, you can also see `coco.py`.


## train
I used the following command to train the models:
```bash
# bisenetv1 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv1_city.py
NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file 

# bisenetv2 cityscapes
export CUDA_VISIBLE_DEVICES=0,1
cfg_file=configs/bisenetv2_city.py
NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file 

# bisenetv1 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3
cfg_file=configs/bisenetv1_coco.py
NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file 

# bisenetv2 cocostuff
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cfg_file=configs/bisenetv2_coco.py
NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file 
```

Note:  
1. though `bisenetv2` has fewer flops, it requires much more training iterations. The the training time of `bisenetv1` is shorter.
2. I used overall batch size of 16 to train all models. Since cocostuff has 171 categories, it requires more memory to train models on it. I split the 16 images into more gpus than 2, as I do with cityscapes.


## finetune from trained model
You can also load the trained model weights and finetune from it, like this:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --finetune-from ./res/model_final.pth --config ./configs/bisenetv2_city.py # or bisenetv1
```


## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --config configs/bisenetv1_city.py --weight-path /path/to/your/weight.pth
```


### Be aware that this is the refactored version of the original codebase. You can go to the `old` directory for original implementation if you need, though I believe you will not need it.


