import os
import argparse

def generate_anno(im_root, split='train'):
    """生成标注文件"""
    im_dir = os.path.join(im_root, 'images', split)
    label_dir = os.path.join(im_root, 'labels', split)
    
    # 获取所有图片
    im_names = [f for f in os.listdir(im_dir) 
                if f.endswith(('.jpg', '.png', '.jpeg'))]
    im_names.sort()
    
    # 生成标注列表
    lines = []
    for im_name in im_names:
        base_name = os.path.splitext(im_name)[0]
        label_name = base_name + '.png'  # 假设标注是png
        
        im_path = os.path.join('images', split, im_name)
        label_path = os.path.join('labels', split, label_name)
        
        # 检查标注是否存在
        full_label_path = os.path.join(im_root, label_path)
        if os.path.exists(full_label_path):
            lines.append(f"{im_path},{label_path}")
        else:
            print(f"Warning: Label not found for {im_name}")
    
    # 保存
    output_file = os.path.join(im_root, f'{split}.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated {output_file} with {len(lines)} samples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_root', default='datasets/water_seg')
    args = parser.parse_args()
    
    generate_anno(args.im_root, 'train')
    generate_anno(args.im_root, 'val')