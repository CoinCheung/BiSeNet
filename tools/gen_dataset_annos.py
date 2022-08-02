
import os
import os.path as osp
import argparse


def gen_coco():
    '''
        root_path:
            |- images
                |- train2017
                |- val2017
            |- labels
                |- train2017
                |- val2017
    '''
    root_path = './datasets/coco'
    save_path = './datasets/coco/'
    for mode in ('train', 'val'):
        im_root = osp.join(root_path, f'images/{mode}2017')
        lb_root = osp.join(root_path, f'labels/{mode}2017')

        ims = os.listdir(im_root)
        lbs = os.listdir(lb_root)

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{mode}2017/{name}.jpg,labels/{mode}2017/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))


def gen_ade20k():
    '''
        root_path:
            |- images
                |- training
                |- validation
            |- annotations
                |- training
                |- validation
    '''
    root_path = './datasets/ade20k/'
    save_path = './datasets/ade20k/'
    folder_map = {'train': 'training', 'val': 'validation'}
    for mode in ('train', 'val'):
        folder = folder_map[mode]
        im_root = osp.join(root_path, f'images/{folder}')
        lb_root = osp.join(root_path, f'annotations/{folder}')

        ims = os.listdir(im_root)
        lbs = os.listdir(lb_root)

        print(len(ims))
        print(len(lbs))

        im_names = [el.replace('.jpg', '') for el in ims]
        lb_names = [el.replace('.png', '') for el in lbs]
        common_names = list(set(im_names) & set(lb_names))

        lines = [
            f'images/{folder}/{name}.jpg,annotations/{folder}/{name}.png'
            for name in common_names
        ]

        with open(f'{save_path}/{mode}.txt', 'w') as fw:
            fw.write('\n'.join(lines))



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', dest='dataset', type=str, default='coco')
    args = parse.parse_args()

    if args.dataset == 'coco':
        gen_coco()
    elif args.dataset == 'ade20k':
        gen_ade20k()
