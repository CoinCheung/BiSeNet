#!/usr/bin/env bash
global_path='./datasets'
data_dir=$global_path
#'/coco'

# mkdir $data_dir
cd $data_dir


git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

echo "Downloading train and validation images"

# Download Images and annotations
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

# Unzip
echo "Unziping train folder"
unzip -q train2017.zip
echo "Unziping val folder"
unzip -q val2017.zip

echo "Deleting zip files"
rm -rf train2017.zip
rm -rf val2017.zip

echo "COCO data downloading over!!"

cd ..
echo "Downloading annotations"
wget -c http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip
unzip -q stuffthingmaps_trainval2017.zip 
rm -rf annotations_trainval2017.zip
rm -rf stuffthingmaps_trainval2017.zip
echo "Done"
