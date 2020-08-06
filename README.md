# BiSeNetV1 & BiSeNetV2

My implementation of [BiSeNetV1](https://arxiv.org/abs/1808.00897) and [BiSeNetV2](https://arxiv.org/abs/1808.00897).


The mIOU evaluation result of the models trained and evaluated on cityscapes train/val set is:
| none | ss | ssc | msf | mscf | fps | link |
|------|:--:|:---:|:---:|:----:|:---:|:----:|
| bisenetv1 | 74.85 | 76.46 | 77.36 | 78.72 | - | [download](https://drive.google.com/file/d/1e1_E7OrpjTaD5Rael7Fus5lg-uGZ5TUZ/view?usp=sharing) |
| bisenetv2 | 74.39 | 74.44 | 76.10 | 75.94 | - | [download](https://drive.google.com/file/d/1r_F-KZg-3s2pPcHRIuHZhZ0DQ0wocudk/view?usp=sharing) |

> Where **ss** means single scale evaluation, **ssc** means single scale crop evaluation, **msf** means multi-scale evaluation with flip augment, and **mscf** means multi-scale crop evaluation with flip evaluation. The eval scales of multi-scales evaluation are `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]`, and the crop size of crop evaluation is `[1024, 1024]`.

Note that the model has a big variance, which means that the results of training for many times would vary within a relatively big margin. For example, if you train bisenetv2 for many times, you will observe that the result of **ss** evaluation of bisenetv2 varies between 72.1-74.4. 


## platform
My platform is like this: 
* ubuntu 16.04
* cuda 10.1.243
* cudnn 7
* miniconda python 3.6.9
* pytorch 1.6.0


## get start
With a pretrained weight, you can run inference on an single image like this: 
```
$ python tools/demo.py --model bisenetv2 --weight-path /path/to/your/weights.pth --img-path ./example.jpg
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

2.custom dataset  

If you want to train on your own dataset, you should generate annotation files first with the format like this: 
```
munster_000002_000019_leftImg8bit.png,munster_000002_000019_gtFine_labelIds.png
frankfurt_000001_079206_leftImg8bit.png,frankfurt_000001_079206_gtFine_labelIds.png
...
```
Each line is a pair of training sample and ground truth image path, which are separated by a single comma `,`.   
Then you need to change the field of `im_root` and `train/val_im_anns` in the configuration files.

## train
In order to train the model, you can run command like this: 
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --model bisenetv2 # or bisenetv1
```

Note that though `bisenetv2` has fewer flops, it requires much more training iterations. The the training time of `bisenetv1` is shorter.


## finetune from trained model
You can also load the trained model weights and finetune from it:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --finetune-from ./res/model_final.pth --model bisenetv2 # or bisenetv1
```


## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --model bisenetv1 --weight-path /path/to/your/weight.pth
```

### Be aware that this is the refactored version of the original codebase. You can go to the `old` directory for original implementation.


