* 看data是否正确，为啥输出变在(360, 640) 了
* 看是否是训练20个类，但是预测时只剩下19个了


#===========
baseline
#===========
1.
momentum = 0.9
weight_decay = 1e-4
lr_start = 1e-3
max_iter = 11000
power = 0.9
warmup_steps = 2000
warmup_start_lr = 1e-6

mIOU = 34.56

2. paper里面的train 参数，
11000iter无warmup: 37.67
21000iter无warmup: 
11000iter无warmup(修正warmup step): 38.18

1000/11000: 38.09


1. 使用auxiliary的loss
aux_loss加3x3conv，使用feat16和32: 35.01
使用8和16的feat加aux_loss

2. ffm的conv_bn_relu的stride改成2: 31.72

3. 最后输出的net，再加一个1x1的conv，stride=1，通道不变: 28.63

4. 训练时只使用其中的19个id，让训练和测试时使用相同的id: 41.97

5. 不加auxiliary: 47.9，看来加auxiliary是有技巧的。

6. 调整sp和cp的输出通道数，让两个均匀一点: 
* sp变成64,256,512: 47.47
* 把cp变成FPN这种: 47.46


7. 训练次数
baseline: 
20类，FPN结构，avg先conv-bn-sig再乘到FPN上面，先interpolate再1x1conv，不加auxiliary
* 11000次: 48.21
* 31000次: 44.12, 48.69, 51.10
* 51000次: 45.48, 46.09, 49.83, 51.17, 52.27
* 71000次: 45.54, 48.50, 48.88, 51.43, 51.53, 52.19, 53.07
* 91000次: 44.53, 47.90, 48.20, 50.03, 51.35, 50.86, 51.74, 52.91, 53.42
看来还是加长训练效果更好。

8. baseline, 51000次, 19类
screen: 37.86, 41.61, 41.98, 44.29, 46.19

9. cp 给interpolate之后再加一个3x3conv
19类, 51000次: 35.86, 40.52, 42.16, 44.25, 46.31 - 加3x3conv有用

9. 都用kaiming_normal给初始化: 
46.22, 48.98, 50.08, 51.59, 52.09 - kaiming_normal作用明显

10. FPN换成nearest
eval里面有interpolate:
47.37, 49.41, 50.67, 52.48, 53.29 - nearest 提高了近一个点
去掉eval里面的interpolate: 左

====
11. 3x3之前的inter变成nearest
eval带inter的: 49.17, 49.33, 53.53, 54.38, 55.27 - 又提高了不少

12. 模型变成20类，把255当成19
eval去掉inter的: 50.48, 53.25, 55.78, 56.18, 57.91 - 看样子是对的

===
13. 使用avg而不是avg_sig:
49.09, 48.90,, 53.72, 55.80, 56.92 - 看样子使用avg_sig比avg好 

14. avg_sig加在feat32上，再跟feat16相加
50.88, 53.48, 54.59, 56.45, 57.97 - 强一丢丢，差不多

15. avg加在feat32上，再跟feat16相加
50.86, 51.83, 54.81, 56.16, 57.50 - 仍然是差不多，总体上，avg_sig就比avg好

==== model.py.bak

16. feat32与avg相乘，再与feat16 concat起来
64, 256, 512, nearest, 无3x3conv: 42.88, 51.15, 51.14, 53.82, 54.98
64, 256, 512, nearest, 有3x3conv, 1280: 45.49, 51.30, 51.69, 53.56, 55.58
64, 256, 512, nearest, 有3x3conv, 1024: 47.60, 51.41, 52.92, 54.22, 55.73
64, 256, 512, nearest, 有1x1conv, 1024: 46.45, 50.23, 51.13, 53.65, 55.19 - 变成1x1不行
64, 256, 1024, nearest, 有3x3conv, 1536: 46.82, 46.12, 50.22, 53.66, 55.42 - 差不多
64, 256, 512, bilinear, 有3x3conv, 1024: 43.04, 48.96, 50.34, 51.71, 53.01 -bilinear不行 

17. feat32与avg_sig相乘，再与feat16 concat起来
64, 256, 512, nearest, 有3x3conv, 1024: 49.68, 51.53, 53.75, 55.56, 56.33 - avg_sig更好

18. 先把feat16和feat32做concat，再与avg-conv-bn-sig相乘
放大都是nearest
后面不加3x3conv, 1280: 48.95, 51.24, 52.39, 54.53, 55.07
后面不加3x3conv, 1792: 47.78, 48.28, 53.26, 53.94, 55.42
后面加3x3conv, 1024: 45.89, 51.59, 52.57, 54.42, 55.27

以上说明concat最好的结果也不如FPN的连接方式
====
用回FPN方式
baseline: 
avg_sig与arm32相乘，再通过FPN与arm16结合，放大成feat8大小，加3x3conv，都用nearest，ffm=512(64,256,512)+512
50.63, 52.60, 55.09, 57.09, 57.98

19. 把sp的conv核加大试试，比如像resnet那种，第一个核改用7x7的
第一个conv改成7x7: 
47.74, 53.10, 54.50, 56.04, 58.22

第一个conv改成3个3x3，第一个s=2, 前两个32,最后一个64:
50.11, 53.87, 54.48, 56.57, 58.10

第二个conv改成5x5的:
50.81, 52.50, 56.01, 56.12, 58.09

第三个conv改成1024, conv3x3: 
50.84, 53.98, 54.62, 57.15, 58.19

第三个conv改成1024, conv5x5: 
50.90, 51.01, 55.43, 56.09, 58.35

- 说明第三个改成1024，并且第一个使用大核效果会更好一些
- 第一个7x7改成3个3x3效果不算好

20. 最后的1x1conv换成3x3的
48.84, 52.52, 55.01, 56.55, 58.31
都差不多，说明作用不大

21. sp使用第一个7x7conv，第二个5x5，第三个5x4 1024。最后的conv使用1x1, 64-256-512
51.56, 54.15, 55.01, 56.34, 58.50

22. cp 换成unsample-8s的仍然是FPN连接
无conv_out: 50.14, 53.07, 55.52, 56.40, 58.21
有conv_out: 49.76, 52.14, 54.78, 56.71, 57.86 - 已经有了conv_outer，再加也没啥用。
无conv_out, FFM的conv改成s=1: 52.41, 55.97, 57.95, 59.40, 60.70 - 作用大
有conv_out, FFM的conv改成s=1: 


23. 最后的feat8是deconv到原图大小
16x16-s8: 51.72, 54.88, 56.85, 59.24, 60.39
改成3个4x4-2，不加bn-relu: 

24. 把cp里面的interpolate换成deconv
inner_conv是FPN形式1x1的: 
47.90, 52.89, 55.58, 57.89, 59.46

inner_conv是FCN形式3x3的，并且没有outer_conv: 
51.39, 50.62, 55.98, 58.44, 60.50 - deconv的话，fcn比fpn结构要好

25. unsample-8s，只upsample就行了，不加上feat8_arm的特征: 
使用FPN结构: screen: 18205
49.09, 54.31, 56.62, 58.22, 59.51

使用FCN结构(kaiming初始化deconv): 
49.84, 52.76, 55.23, 56.86, 59.50 (baseline)

- 看来有必要加上feat8_arm的特征


26. FFM不改变输入的channel数，FFM之后加上1x1conv调通道再upconv
screen: 18205

27. 去掉conv16_outer直接输出特征之和
48.97, 53.31, 55.92, 57.77, 59.90


27. avg当成fcn的一层，放大然后加上去
去掉conv16_outer之后:
48.82, 53.12, 55.82, 57.51, 59.74 


25. 加上auxiliary loss
只加一个feat32的: 
一次放大32倍


22. feat32 deconv成feat8, feat16 deconv成feat8，然后element加成feat8那么大


22. cp的最后改成avg_pool然后unsample，再加到feat32上去


13. sp改成32,64,256，调整sp和cp的channel数比例

14. 最基本的原始inter + concat方法，放弃conv

15. 看两张卡跟一张卡比较的结果



7. 使用90k个iter去训练
* cp变成FPN(bilinear), sp 64, 256, 512:
54.20 - 提高个点
* cp变成FPN(nearest), sp 64, 256, 512: 晚
* cp先FPN再channel-atten相乘(avg做conv-bn-sigmoid再乘): 更晚

* 先interpolate到原图大小，再conv1x1


* 对arm输出先conv-bn-relu再interpolate，再1x1conv再用auxliary_loss: 


* interpolate之后再加上conv-bn-relu


7. 使用自己的训练sheduel，两个step这种



2. eval 时像论文那样使用1024x1024

2. 如果不收敛就先warmup一下再弄



===================
1. 论文中的 baseline: 
使用64-32-16的大核作为out + 不使用auxiliary
49.17, 52.17, 54.44, 56.53, 57.64

加上auxiliary loss:
都16和32deconv成8s，然后再放大到原图，主out没有加deconv，直接用了双线性。
46.44, 51.08, 52.13, 54.90, 55.94

不加auxiliary, feat32先3x3conv到n_classes，再interpolate，最后1x1conv
52.32

2. 不加auxiliary, feat32先1x1conv到n_classes，再interpolate，最后3x3conv, 改成90k
23.81, 34.05, 37.19, 40.61, 42.23, 43.49, 45.99, 46.58, 48.06

3. 像psp那样，先conv3x3(512)-bn-relu，再1x1conv(n_classes)，再interpolate
52.65, 53.59, 54.19, 55.32, 55.81, 57.03, 55.78, 58.02, 58.50


=====
加u-shape
1. 两个都是1x1conv-bn-relu，再concat




==============
baseline: cropsize=1024x1024
65.64

+ auxiliary: PSP那种，一个conv-bn-relu加一个1x1conv
2048x1024: 65.69
1536x768: 62.71

+ context 输出改成256通道
2048x1024: 66.59

+ auxiliary 加上dropout 0.1
66.73

+ linear 改成 nearest
67.41

+ sp 改成48
右

+ 加上global avg

+ feat_out之前加上一个conv-bn-relu

+ 大cropsize，bn

+ cropsize改成713\*713，使用gn


======
baseline:
feat32直接加out_head: 56.54
feat32先inter到8，再加out_head: 61.13
把conv-bn-relu-dropout(out_chan=256)当成fcnhead: 61.25

baseline的做法是:
* feat32先nearest, upsample到8，再加conv-bn-relu(512-256)，然后再conv_out到n_classes
* 使用fcn的head放在feat16上当成auxiliary loss


加u-shape:
feat32先upsample到16，再与feat16concat起来，再加conv-bn-relu到512，upsample到8, 
最后fcnhead，feat32和feat16都加上fcnhead做成auxiliary loss
68.08


加sp
* 用sum:
64-128-512，再相加，然后正常fcn_head_out:  68.76
64-256-512，再相加，然后正常fcn_head_out:  68.79
32-128-512，再相加，然后正常fcn_head_out:  68.59
32-512-512: 68.76
64-512-512: 68.83
128-512-512: 68.88
128-256-512: 68.72
512-512-512:  (memory overflow)

用sum应该比ushape提高0.8个点,

* 用ffm
sp用的32-128-512
512+512cat到1024之后，conv成1024: 69.25
512+512cat到1024之后，conv成512: 69.13

conv成1024:
sp 用64-256-512: 69.10
sp 用64-512-512: 
sp 用128-512-512: 
sp 用256-512-512: 


用ffm应该比ushape提高1.4个点。  

final: 
sp: 32-128-512: 69.127


加arm不加avg: 
arm学习特征: 
* arm加在aux loss之前(aux loss不考虑arm)
68.98
* aux loss考虑arm
68.02

arm学习残差: 
* arm参与aux loss
68.28
使用自制resnet: 
68.30

* arm不参与aux loss
68.91

arm直接合并到bottleneck里面，就像se-resnet一样
67.84

arm 合到bottleneck里面, 不加bn:
67.62

arm合到bottleneck里面，改成se的结构conv-relu-con-sig
67.40

用arm应该比ffm提高1.3


加avg不加arm:
加avg应该会比ffm提高1个点
68.68
让avg变成atten，乘到feat32上去: 
68.81

让avg变成atten，加到feat32上去: 
68.73

把iter数减小到40k试试: 
arm和avg都不要: 66.95
加上resnet之外的arm: 67.01

iter减小到70k
加上resnet之外的arm: 68.64

使用修改后的scale的aug

avg先1x1conv之后，再bn再加到feat32上，不考虑sigmoid，如果不是相乘的话。

arm改成conv-bn-relu，不使用conv-bn-sig

去掉warmup，去掉scale，使用标准xception



使用TrochSeg的结构: 
nearest:  
bilinear:  62.21，比不带inplace和dist的小0.3个点，可能是inplace的moment这一些的问题

学习率改成2.5e-2: 
61.30

加上hard-mining: 
bilinear, thresh=0.7, n_min=n_pixs/world_size/16

lr=1e-2, bilinear + hard mining:
cropsize=1536x768: 73.05 <-- 到这里了
cropsize=2048x1024: 73.10
flip+msc: 75.32

试一下crop eval. scale单是1x1的，没有放大scale,单是对原图crop eval

hard-mining, nearest, 不tune pretrained的bn, wd=5e-4:  
flip+msc:
74.87
no flip+msc: 73.51

学习率按照卡数减小试试。 
76.07 - 说明之前是lr太大了. 

新加的layer用10倍的lr
75.13

使用bilinear不用nearest: 
74.66

重新弄lrx10的，把optimizer里面也改改，
81k:
    lr0=1e-2
        nearest: 75.86
        bilinear: 
    lr0=5e-3
        nearest: 
        bilinear: 
80epoch: 
    lr0=1e-2
        nearest: 74.23
        bilinear: 73.46
    lr0=5e-3
        nearest: 73.91
        bilinear: 73.42
还得用80k，原实现也是80epoch x 1000iter_per_epoch的

看一下torchseg里面的epoch怎么算的，看一下msc是否应该是logits相加，看一下flip是水平flip还是竖直flip:
这个原实现用的是先把flip的logits相加，再做torch.exp，把各个scale的exp再相加，而且使用的是crop 1024的方法做的。 

不加msc+flip应该是76.2才对.  

1e-2, nearest不加warmup试试: 71.97



1e-2, 80k, 1warmup
ffm 和 BiseNetOutput使用10倍lr，其他都是1倍的lr, 所有的bn和bias都不加wd，bilinear
bilinear: 75.65
    flip加exp: 75.80
    crop evaluation: empty: 77.42(stride=2/3), 77.42(stride=5/6), zeros: 77.42
nearest: 75.38

bilinear: 2e-2
76.35
bilinear: 5e-3
77.16

加上warmup的, bilinear, lr=1e-2, 80k: 
cropsize = 1024
78.24
cropsize = 960，跟训练时一样大:
78.16
说明eval的时候还是大一点的crop size比较好，跟训练不一致也没关系，只要别大的离谱。 

lr=1e-2, wd=5e-4, 80k, warmup=1000, hard-mining, color_jitter, crop=960:
78.90

尝试新的更省内存的ohem，与上面结果是否一致: 
78.29 - 好像又坏了。 

新ohem, 用回1024做train试试: 
78.23 - 好像没有改变

再试旧的ohem+960: 
78.13

试试deeplab的ohem，960不够大了，算了

final: crop=1024, 新的ohem

使用nearest试试: 
78.63


refactoring: 



试试focal loss: 

使用res2代替sp: 

