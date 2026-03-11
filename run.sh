###################################### 数据集划分 #####################################
#数据集分割 train 和 val
python ./tools/auto_split_dataset.py

# 默认划分：80%训练，20%验证
python tools/auto_split_dataset.py --data-root datasets/water_seg
# 自定义比例：90%训练，10%验证
python tools/auto_split_dataset.py --data-root datasets/water_seg --train-ratio 0.9
# 自定义随机种子（保证划分结果可复现）
python tools/auto_split_dataset.py --data-root datasets/water_seg --seed 123

# 恢复原状（如果划分错了）
python tools/auto_split_dataset.py --data-root datasets/water_seg --restore

###################################### 数据集标注 #####################################
#数据集标注，生成 txt 文件
python ./tools/gen_water_annos.py

###################################### 模型训练 #####################################
#模型训练
$env:CUDA_VISIBLE_DEVICES="0"
python tools/train_amp_water.py --config configs/bisenet_water.py


#模型测试
python tools/demo.py --config configs/bisenet_water.py --weight-path ./res/model_final.pth --img-path ./test_water.jpg
