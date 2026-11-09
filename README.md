# BiSeNetV1 & BiSeNetV2

My implementation of [BiSeNetV1](https://arxiv.org/abs/1808.00897) and [BiSeNetV2](https://arxiv.org/abs/2004.02147).


mIOUs and fps on cityscapes val set:
| none | ss | ssc | msf | mscf | fps(fp32/fp16/int8) | link |
|------|:--:|:---:|:---:|:----:|:---:|:----:|
| bisenetv1 | 74.74 | 76.12 | 76.90 | 78.36 | 71/165/426 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v1_city.pth) |
| bisenetv2 | 74.52 | 75.55 | 76.49 | 77.16 | 72/171/108 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v2_city.pth) |

mIOUs on cocostuff val2017 set:
| none | ss | ssc | msf | mscf | link |
|------|:--:|:---:|:---:|:----:|:----:|
| bisenetv1 | 34.30 | 34.20 | 35.43 | 35.49 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v1_coco.pth) |
| bisenetv2 | 29.92 | 29.74 | 31.08 | 30.81 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v2_coco.pth) |

mIOUs on ade20k val set:
| none | ss | ssc | msf | mscf | link |
|------|:--:|:---:|:---:|:----:|:----:|
| bisenetv1 | 35.86 | 35.70 | 37.20 | 36.65 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v1_ade20k.pth) |
| bisenetv2 | 33.42 | 33.16 | 34.44 | 33.49 | [download](https://github.com/CoinCheung/BiSeNet/releases/download/0.1.0/model_final_v2_ade20k.pth) |

Tips: 

1. **ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. The eval scales and crop size of multi-scales evaluation can be found in [configs](./configs/).

2. The fps is tested in different way from the paper. For more information, please see [here](./tensorrt).

3. The authors of bisenetv2 used cocostuff-10k, while I used cocostuff-123k(do not know how to say, just same 118k train and 5k val images as object detection). Thus the results maybe different from paper. 

4. The model has a big variance, which means that the results of training for many times would vary within a relatively big margin. For example, if you train bisenetv2 on cityscapes for many times, you will observe that the result of **ss** evaluation of bisenetv2 varies between 73.1-75.1. Thus the numbers above are the mean of many runs: 32 runs for cocostuff and bisenetv2 on ade20k, and 8 runs for the others. 


## deploy trained models

1. tensorrt  
You can go to [tensorrt](./tensorrt) for details.  

2. ncnn  
You can go to [ncnn](./ncnn) for details.  

3. openvino  
You can go to [openvino](./openvino) for details.  

4. tis  
Triton Inference Server(TIS) provides a service solution of deployment. You can go to [tis](./tis) for details.


## platform

My platform is like this: 

* ubuntu 24.04
* nvidia A40 gpu
* cuda 12.6
* cudnn 9
* miniconda python 3.12
* pytorch 2.12


## get start

With a pretrained weight, you can run inference on an single image like this: 

```
$ python tools/demo.py --config configs/bisenetv2_city.py --weight-path /path/to/your/weights.pth --img-path ./example.png
```

This would run inference on the image and save the result image to `./res.jpg`.  

Or you can run inference on a video like this:  
```
$ python tools/demo_video.py --config configs/bisenetv2_coco.py --weight-path res/model_final.pth --input ./video.mp4 --output res.mp4
```
This would generate segmentation file as `res.mp4`. If you want to read from camera, you can set `--input camera_id` rather than `input ./video.mp4`.   


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
$ python tools/gen_dataset_annos.py --dataset coco
```

3.ade20k

Download `ADEChallengeData2016.zip` from this [website](http://sceneparsing.csail.mit.edu/) and unzip it. Then we can move the uncompressed folders to `datasets/ade20k`, and generate the txt files with the script I prepared for you:  
```
$ unzip ADEChallengeData2016.zip
$ mv ADEChallengeData2016/images /path/to/BiSeNet/datasets/ade20k/
$ mv ADEChallengeData2016/annotations /path/to/BiSeNet/datasets/ade20k/
$ python tools/gen_dataset_annos.py --dataset ade20k
```


4.custom dataset  

If you want to train on your own dataset, you should generate annotation files first with the format like this: 
```
munster_000002_000019_leftImg8bit.png,munster_000002_000019_gtFine_labelIds.png
frankfurt_000001_079206_leftImg8bit.png,frankfurt_000001_079206_gtFine_labelIds.png
...
```
Each line is a pair of training sample and ground truth image path, which are separated by a single comma `,`.   

I recommand you to check the information of your dataset with the script:  
```
$ python tools/check_dataset_info.py --im_root /path/to/your/data_root --im_anns /path/to/your/anno_file
```
This will print some of the information of your dataset.  

Then you need to change the field of `im_root` and `train/val_im_anns` in the config file. I prepared a demo config file for you named [`bisenet_customer.py`](./configs/bisenet_customer.py). You can start from this conig file.


## train

Training commands I used to train the models can be found in [here](./dist_train.sh).

Note: I trained all models on a single gpu, with batch size of 16 on cityscapes, and 32 on cocostuff and ade20k.


## finetune from trained model

You can also load the trained model weights and finetune from it, like this:
```
$ export CUDA_VISIBLE_DEVICES=0
$ torchrun --nproc_per_node=1 tools/train_amp.py --finetune-from ./res/model_final.pth --config ./configs/bisenetv2_city.py # or bisenetv1
```


## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --config configs/bisenetv1_city.py --weight-path /path/to/your/weight.pth
```
or you can use multi gpus:  
```
$ torchrun --nproc_per_node=2 tools/evaluate.py --config configs/bisenetv1_city.py --weight-path /path/to/your/weight.pth
```


### Be aware that this is the refactored version of the original codebase. You can go to the `old` directory for original implementation if you need, though I believe you will not need it.


