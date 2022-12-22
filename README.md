# FISH-SPECIES-DETECTION

## Aim and Objectives

# Aim

• To create a Fish species detection system which will detect objects based on whether it is Shark fish, Star fish or Cat fish.

## Objectives
• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.
    
• Using appropriate datasets for recognizing and interpreting data using machine learning.
    
• To show on the optical viewfinder of the camera module whether objects are Star fish, shark or cat fish.

## Abstract

• An object is classified based on whether it is Shark , cat fish or star fish.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.
    
• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.
    
• Over the last few years, several research works have been performed to monitor fish in the underwater environment aimed for marine research, understanding ocean geography, and primarily for sustainable fisheries. 
    
• Automating fish identification is very helpful, considering the time and cost of the manual process. However, it can be challenging to differentiate fish from the seabed and fish types from each other due to environmental challenges like low illumination, complex background, high variation in luminosity, free movement of fish, and high diversity of fish species. 

## Introduction

• This project is based on a Fish species detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• This project can also be used to gather information about what species of fish does the object comes in.
    
• The objects can even be further classified into star fish, cat fish and shark based on the image annotation we give in roboflow.

• However, it can be challenging to differentiate fish from the seabed and fish types from each other due to environmental challenges like low illumination, complex background, high variation in luminosity, free movement of fish, and high diversity of fish species and gets harder for the model to detect. However, training in Roboflow has allowed us to crop images and also change the contrast of certain images to match the time of day for better recognition by the model.
    
• Neural networks and machine learning have been used for these tasks and have obtained good results.
    
• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Fish species detection as well.
    
## Literature Review

• Today, underwater fish detection is in high demand for different purposes, such as research in marine science and oceanography and monitoring aquaculture for sustainable fisheries..
    
• In 2018, total global capture of fisheries production reached the highest level ever recorded at 96.4 million tonnes. Most of the production comes from captured marine fisheries, 84.4 million tonnes.
    
• Real-time monitoring of those commercial fisheries will help produce more and match human consumption demand. 
    
• Underwater videos and images offer a non-intrusive, cost-effective way to collect large volumes of visual data to process the information. However, manual processing of underwater videos and images is labor-intensive, time-consuming, expensive, and prone to fatigue errors. 
    
• Therefore, the automatic processing of underwater videos for fish detection is an attractive alternative. However, the unrestricted environmental factors such as complex background, luminosity, camouflage foreground, crowded and dynamic background, and so on make the task challenging. These influences compromise the accuracy of fish detection. 
    
• Besides that, traditional automatic approaches can not detect underwater fish with reasonable detection rates due to the illumination changes.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.
   
• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Jetson Nano 2GB:-

![nano_img01](https://user-images.githubusercontent.com/93208224/209101765-3af37ed9-bb99-4370-a340-0d1f442709d0.jpg)






## Proposed System
    
• Study basics of machine learning and image recognition.

• Start with implementation

 ➢ Front-end development
 ➢ Back-end development
    
• Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether objects are star fish, cat fish or shark.
    
• Use datasets to interpret the object and suggest whether the object is cat fish, star fish or shark.

## Methodology 
    
• The Fish Species detection system is a program that focuses on implementing real time Fish species detection.
    
• It is a prototype of a new product that comprises of the main module: Fish detection and then showing on viewfinder whether the object is Cat fish, 
star fish or shark.

• Fish Species Detection Module

This Module is divided into two parts:

1] Fish detection
        
1. Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.
        
2. For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
        
3. This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.
        
2] Classification Detection
    
1. Classification of the object based on whether it is Shark, cat fish or star fish.
    
2. Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.
    
3. There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
    
4. YOLOv5 was used to train and test our model for various classes like star fish , cat fish or shark. We trained it for 100 epochs and achieved an accuracy of approximately 91%.

## Installation

### Initial Setup
Remove unwanted Applications.
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
Create Swap file
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
#################add line###########
/swapfile1 swap swap defaults 0 0
Cuda Configuration
vim ~/.bashrc
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
source ~/.bashrc
Udpade a System
sudo apt-get update && sudo apt-get upgrade
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
source ~/.bashrc
sudo pip3 install pillow
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo python3 -c "import torch; print(torch.cuda.is_available())"
Installation of torchvision.
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
Clone yolov5 Repositories and make it Compatible with Jetson Nano.
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0

### Fish Species Dataset Training
   
   • We used Google Colab And Roboflow
   
   • Train your model on colab and download the weights and paste them into yolov5 folder link of project

Running Fish Species Detection Model
source '0' for webcam
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

## Demo



 
## Advantages
 
  High accuracy of fish detection
 
  Less human intervention

## Future Scope
   
 • As we know technology is marching towards automation, so this project is one of the step towards automation.
    
 • Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
    
 • Garbage segregation will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.
    
 • As more products gets released due to globalization and urbanization new waste will be created and hence our model which can be trained and modified with just the addition of images can be very useful.
 
## Conclusion

• In this project our model is trying to detect objects and then showing it on viewfinder, live as what their class is as whether they are star fish, cat fish or shark as we have specified in Roboflow.
    
• Thus the obtained datasets were preprocessed by using deep learning and yolov5 as tool to detect the fishes in the datasets were detected and the accuracy was displayed. The type of fishes were displayed. 
    
• This fish recognition and detection can be used to detect the fishes, recognise the species in the water and also for commercial purposes.
    
• Being implemented in real time it can be enhanced by use of high end underwater cameras to get more accuracy in the output.

## References

• Roboflow :- https://roboflow.com/
    
• Datasets or images used: https://www.gettyimages.ae/search/2/image?phrase=fish

• Google images
