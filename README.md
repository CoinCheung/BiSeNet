# BiSeNet
My implementation of [BiSeNet](https://arxiv.org/abs/1808.00897). My environment is pytorch1.0 and python3, the code is not tested with other environments, but it should also work on similiar environments.


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
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
This would take almost one day on a two 1080ti gpu, and the mIOU will also be computed after training is done.  
You can also run the evaluate alone script after training:  
```
    $ python evaluate.py
```

Currently, I achieved mIOU of 78.6.



### Tricks:  
These are the tricks that I find might be useful:  
1. use online hard example ming loss. This let the model be trained more efficiently.  
2. do not add weight decay when bn parameters and bias parameters of nn.Conv2d or nn.Linear are tuned.  
3. use a 10 times larger lr at the model output layers.  
4. use crop evaluation. We do not want the eval scale to be too far away from the train scale, so we crop the chips from the images to do evaluation and then combine the results to make the final prediction.  
5. multi-scale training and multi-scale-flip evaluating. On each scale, the scores of the original image and its flipped version are summed up, and then the exponential of the sum is computed to be the prediction of this scale.   
6. warmup of 1000 iters to make sure the model better initialized.   


