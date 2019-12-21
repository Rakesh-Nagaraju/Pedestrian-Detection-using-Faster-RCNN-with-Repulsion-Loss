# Repulsion Loss

This repository implements the code for [Repulsion Loss: Detecting pedestrians in a crowd]([https://arxiv.org/pdf/1711.07752v2.pdf](https://arxiv.org/pdf/1711.07752v2.pdf)). This page(code, project and presentation) is the submission for Group B for final project for the course CS 256: Topics in Artificial Intelligence, Section 2 led by Prof. Mashhour Soul, PhD.
The group members include: 
* [Vidish Naik](mailto:vidish.naik@sjsu.edu)
* [Joel Alvares](mailto:joel.alvares@sjsu.edu)
* [Charulata Lodha](mailto:charulata.lodha@sjsu.edu)
* [Rakesh Nagaraju](mailto:rakesh.nagaraju@sjsu.edu)

The code maybe used for educational and commercial use under no warranties. 
For questions on this project and code please reach out to: `Vidish Naik` at `vidish.naik@sjsu.edu`

# Requirements:
* PyTorch: 0.4.0
* TorchVision: 0.2.0
* CUDA: 8.0

# Preparation

Clone this repository
```sh
git clone https://github.com/VidishNaik/RepulsionLoss.git
```
cd into the folder
```sh
cd RepulsionLoss
```
Export the path for CUDA:
```sh
export CUDA_HOME="/usr/local/cuda-8.0"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64":$LD_LIBRARY_PATH
export PATH="/usr/local/cuda-8.0/bin:$PATH"
```
Verify that CUDA is indeed set to CUDA8
```sh
nvcc -V
```
>nvcc: NVIDIA (R) Cuda compiler driver
>Copyright (c) 2005-2016 NVIDIA Corporation
>Built on Tue_Jan_10_13:22:03_CST_2017
>Cuda compilation tools, release 8.0, V8.0.61

Install the requirements
```sh
pip install -r requirements.txt
```
Run the `startup.sh` script
```sh
sh startup.sh
```
This will compile the required libraries in the lib folder and also download **PASCAL VOC dataset** in the dataset folder as well as download a pre-trained model in **data/pretrained_model** directory.

**NOTE:** The libraries in the current state are compiled for Nvidia Tesla K80 GPU. Change the CUDA_ARCH argument in `lib/make.sh` if you're using a different GPU. Refer this [link]([https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)) for the correct architecture for your GPU. You can also refer the [CUDA WIKI]([https://en.wikipedia.org/wiki/CUDA#GPUs_supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)) to get the correct architecture. 

# Running the model
```sh
python trainval_net.py --dataset pascal_voc --net vgg16 --bs 4 --nw 1 --cuda
```
* Keep batch size (bs) < 4 on a Tesla K80 GPU as it will run out of memory
* Currently only works for `vgg16`. Future updates might include resnet101.
* dataset can be changed to pascal_voc_0712 for PASCAL VOC 2012

# Model download
For testing the model [download](https://rep-loss-model.s3.amazonaws.com/faster_rcnn_1_10_2504.pth) the pre-trained weights and store it in `models/vgg16/pascal_voc/faster_rcnn_1_10_2504.pth`
# Testing
```sh
python test_net.py --net vgg16 --checksession 1 --checkepoch 10 --checkpoint 2504 --cuda
```
# Demo
Store the images that you need to run in the images folder. New images with detections will be stored in the same folder
```sh
python demo.py --cuda --load_dir models --net vgg16 --checksession 1 --checkepoch 10 --checkpoint 2504
```
# Credits
This project was conducted with free credits provided by AWS educate team.
