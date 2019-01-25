# BiSeNet
My implementation of [BiSeNet](https://arxiv.org/abs/1808.00897). My environment is pytorch1.0 and python3, the code is not tested with other environments.


### get cityscapes dataset
Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them in the `data/` directory:  
```
    $ mkdir -p data
    $ mv /path/to/leftImg8bit_trainvaltest.zip data
    $ mv /path/to/gtFine_trainvaltest.zip data
    $ cd data
    $ unzip leftImg8bit_trainvaltest.zip
    $ unzip gtFine_trainvaltest.zip
```

### train and evaluation
Just run the train script: 
```
    $ python train.py
```
This would take almost one day on a single 1080ti gpu, and the mIOU will also be computed after training is done.



### Notes:  
The paper proposes two versions with different backbones: resnet18 and xception39. I only implement the resnet18 version. There are a lot of results reported in the paper. I only implement the final experiments in the ablation study. The target mIOU is 71.4 and the crop size is (640, 360). 

Since the paper does not mention the training iters, I simply used a 9k schema. After plenty of experiments, I got to find that: With merely `u-shape+sp+ffm`, the mIOU can get the value of 69.1, much higher than reported in the paper(guess this is because of the difference between resnet18 backbone and xception39 backbone). Notwithstanding, with additional `ARM` and `GP`, the mIOU would drops to the range of 67-69.

I have tried various ways to use these two modules, include using `GP` as some attention vector to be multiplied to the feature (instead of directly addition) and adding `ARM` to the resnet residual path rather than to the residual block output as does with [SENET](https://arxiv.org/abs/1709.01507). These methods all failed, so I did not involve them in this repository.

In a word, I am stuck, and I cannot make further improvement with `ARM` or `GP`. Please spare some light on me, if you have better understanding of the usage of these two modules.
