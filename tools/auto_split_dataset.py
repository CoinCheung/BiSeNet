import os
import shutil
import random
import argparse
from pathlib import Path


def auto_split_dataset(data_root, train_ratio=0.8, seed=42):
    """
    自动划分数据集为train和val
    
    Args:
        data_root: 数据集根目录，包含 images/ 和 labels/ 文件夹
        train_ratio: 训练集比例（默认0.8 = 80%训练，20%验证）
        seed: 随机种子，保证可复现
    """
    random.seed(seed)
    
    # 路径设置
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    labels_dir = data_root / 'labels'
    
    # 检查文件夹是否存在
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到文件夹: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到文件夹: {labels_dir}")
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 匹配标签
    valid_pairs = []
    for img_path in image_files:
        # 尝试找到对应的标签文件
        # 优先级: .png > .jpg > 同名不同后缀
        label_path = labels_dir / (img_path.stem + '.png')
        if not label_path.exists():
            label_path = labels_dir / (img_path.stem + '.jpg')
        if not label_path.exists():
            label_path = labels_dir / img_path.name
        
        if label_path.exists():
            valid_pairs.append((img_path.name, label_path.name))
        else:
            print(f"警告: 找不到标签文件 for {img_path.name}")
    
    print(f"成功匹配 {len(valid_pairs)} 对数据")
    
    if len(valid_pairs) == 0:
        raise ValueError("没有有效的图片-标签对！")
    
    # 随机打乱并划分
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"划分结果: 训练集 {len(train_pairs)} 张, 验证集 {len(val_pairs)} 张")
    
    # 创建新的目录结构
    # 先备份原文件夹
    backup_images = data_root / 'images_backup'
    backup_labels = data_root / 'labels_backup'
    
    if not backup_images.exists():
        shutil.move(str(images_dir), str(backup_images))
        print(f"原images文件夹已备份到: {backup_images}")
    if not backup_labels.exists():
        shutil.move(str(labels_dir), str(backup_labels))
        print(f"原labels文件夹已备份到: {backup_labels}")
    
    # 创建新的目录结构
    new_images_dir = data_root / 'images'
    new_labels_dir = data_root / 'labels'
    
    (new_images_dir / 'train').mkdir(parents=True, exist_ok=True)
    (new_images_dir / 'val').mkdir(parents=True, exist_ok=True)
    (new_labels_dir / 'train').mkdir(parents=True, exist_ok=True)
    (new_labels_dir / 'val').mkdir(parents=True, exist_ok=True)
    
    # 复制训练集
    print("复制训练集...")
    for img_name, label_name in train_pairs:
        shutil.copy(str(backup_images / img_name), 
                   str(new_images_dir / 'train' / img_name))
        shutil.copy(str(backup_labels / label_name), 
                   str(new_labels_dir / 'train' / label_name))
    
    # 复制验证集
    print("复制验证集...")
    for img_name, label_name in val_pairs:
        shutil.copy(str(backup_images / img_name), 
                   str(new_images_dir / 'val' / img_name))
        shutil.copy(str(backup_labels / label_name), 
                   str(new_labels_dir / 'val' / label_name))
    
    # 生成标注文件
    generate_anno_files(data_root, train_pairs, val_pairs)
    
    print("\n✅ 数据集划分完成！")
    print(f"新目录结构:")
    print(f"  {data_root}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({len(train_pairs)} 张)")
    print(f"  │   └── val/ ({len(val_pairs)} 张)")
    print(f"  ├── labels/")
    print(f"  │   ├── train/ ({len(train_pairs)} 张)")
    print(f"  │   └── val/ ({len(val_pairs)} 张)")
    print(f"  ├── train.txt")
    print(f"  ├── val.txt")
    print(f"  ├── images_backup/ (原数据备份)")
    print(f"  └── labels_backup/ (原数据备份)")


def generate_anno_files(data_root, train_pairs, val_pairs):
    """生成train.txt和val.txt标注文件"""
    data_root = Path(data_root)
    
    # 训练集标注
    with open(data_root / 'train.txt', 'w') as f:
        lines = []
        for img_name, label_name in train_pairs:
            # 统一使用png作为标签后缀（方便后续处理）
            img_stem = Path(img_name).stem
            lines.append(f"images/train/{img_name},labels/train/{img_stem}.png")
        f.write('\n'.join(lines))
    
    # 验证集标注
    with open(data_root / 'val.txt', 'w') as f:
        lines = []
        for img_name, label_name in val_pairs:
            img_stem = Path(img_name).stem
            lines.append(f"images/val/{img_name},labels/val/{img_stem}.png")
        f.write('\n'.join(lines))
    
    print(f"已生成: {data_root}/train.txt 和 {data_root}/val.txt")


def restore_backup(data_root):
    """恢复原数据（如果划分错了）"""
    data_root = Path(data_root)
    backup_images = data_root / 'images_backup'
    backup_labels = data_root / 'labels_backup'
    images_dir = data_root / 'images'
    labels_dir = data_root / 'labels'
    
    if backup_images.exists():
        if images_dir.exists():
            shutil.rmtree(images_dir)
        shutil.move(str(backup_images), str(images_dir))
        print("已恢复 images 文件夹")
    
    if backup_labels.exists():
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
        shutil.move(str(backup_labels), str(labels_dir))
        print("已恢复 labels 文件夹")
    
    # 删除生成的txt文件
    for f in ['train.txt', 'val.txt']:
        if (data_root / f).exists():
            (data_root / f).unlink()
    
    print("✅ 已恢复原状")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自动划分数据集为train/val')
    parser.add_argument('--data-root', default='datasets/water_seg',
                       help='数据集根目录（包含images和labels文件夹）')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8 = 80%）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--restore', action='store_true',
                       help='恢复原数据（撤销划分）')
    
    args = parser.parse_args()
    
    if args.restore:
        restore_backup(args.data_root)
    else:
        auto_split_dataset(args.data_root, args.train_ratio, args.seed)