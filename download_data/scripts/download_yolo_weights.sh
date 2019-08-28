#!/bin/bash
mkdir weights

# Download weights for vanilla YOLOv3
wget -c -P ./weights https://pjreddie.com/media/files/yolov3.weights
# # Download weights for tiny YOLOv3
wget -c -P ./weights https://pjreddie.com/media/files/yolov3-tiny.weights
# Download weights for backbone network (Darknet)
wget -c -P ./weights https://pjreddie.com/media/files/darknet53.conv.74
# Download weights for tiny Darknet
wget -c -P ./weights https://pjreddie.com/media/files/tiny.weights
