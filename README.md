# Object-detection-SNPE
Object detection application using SNPE (Snapdragon Neural Processing Engine)

## What is this?
This application is made for testing the speed and accuracy of object detection deep learning model.

This appliation supports two mode: camera and image.  
Camera mode runs object detection from camera input and displays the result on monitor. This shows the speed and accuracy of model intuitively and visually.  
Image mode runs object detection with given image files and prints the average inference speed and mAP (mean absolute precision). This shows the speed and accuracy of model quantitatively.  

## Build
This application runs only on qualcomm HW since it uses SNPE.  

SNPE, opencv (only core and imgproc), and qcarcam client libraries for aarch64 Linux is included. If your system is not aarch64 Linux, you will need these libraries.  
Qcarcam (ais_client), fastcv, json-c, jpeg libraries are additionally required.  
Python3.x and python modules (collections, numpy, math, json) are requried for calculating mAP.  

After checking theses dependencis, build app using make.  
It builds pkshin_detect binary.  

## How to use

Type pkshin_detect --help to show guide.  

run_camea.sh is an example running camera mode.  
run_mAP.sh is an example running image mode and calculating mAP.  

SSD Mobilenet V1 and V2, COCO2017 validation data set and annotations are included.  
Go to model directory to see how I converted the model. 

You can of course change the model file and data set by yourself.

## Result

It was tested on SA8195 running Automotive Grade Linux.  

![image](https://user-images.githubusercontent.com/28533445/149467971-f3af9c78-72fd-4f88-81c6-318ef7b42275.png)
