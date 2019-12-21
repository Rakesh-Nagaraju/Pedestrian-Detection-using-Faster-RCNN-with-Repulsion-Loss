#!/bin/bash
cd lib
sh make.sh
cd ..
mkdir data
cd data
sh voc2007.sh
mv VOCdevkit/ VOCdevkit2007/
mkdir pretrained_model
cd pretrained_model
wget https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth

