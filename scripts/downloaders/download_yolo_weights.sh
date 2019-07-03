#!/bin/bash
mkdir weights

# Download weights for vanilla YOLOv3
wget -c -P ./weights https://pjreddie.com/media/files/yolov3.weights
# # Download weights for tiny YOLOv3
wget -c -P ./weights https://pjreddie.com/media/files/yolov3-tiny.weights
# Download weights for backbone network
wget -c -P ./weights https://pjreddie.com/media/files/darknet53.conv.74
