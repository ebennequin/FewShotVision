# This file needs to be run from project root folder

#!/usr/bin/env bash

mkdir -p data/miniImageNet
cd data/miniImageNet

wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/train.csv
wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/val.csv
wget https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/test.csv
wget http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz
tar -zxvf ILSVRC2015_CLS-LOC.tar.gz

cd ../..

python src/downloaders/miniImageNet/write_mini_imagenet_filelist.py
python src/downloaders/miniImageNet/write_cross_filelist.py
