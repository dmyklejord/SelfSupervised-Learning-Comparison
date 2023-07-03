# SelfSupervised-Learning-Comparison
A script for looping through and training different self-supervised learning algorithms.


| SimCLR | MoCo | BYOL |
| :---: | :---: | :--: |
| ![SimCLR](ReadmeImages/Screenshot%202023-03-03%20at%204.19.39%20PM.png) | ![MoCo](ReadmeImages/Screenshot%202023-03-03%20at%204.22.56%20PM.png) | ![BYOL](ReadmeImages/Screenshot%202023-03-03%20at%204.23.55%20PM.png) |

\* *Figures are from their respective papers.*

## What this script does
This script investigates the effect of using less data for training the SSL models, and thus loops through using 5, 10, 20, 50, 100, and 200 datapoints per class (and then uses all the data as a baseline). It also tries 8 different learning with each training set size. This whole process is done twice, once with SimCLR-paper augmentations, and again with much weaker augmentations. Accuracy is judged using the linear classification protocol.

You'll end up with a folder like this for each iteration:
![ResultingFiles](ReadmeImages/Screenshot%202023-07-03%20at%208.30.16%20AM.png)

## Compiled Results:
![FullResults1](ReadmeImages/Screenshot%202023-07-03%20at%209.00.44%20AM.png)
![FullResults2](ReadmeImages/Screenshot%202023-07-03%20at%209.00.54%20AM.png)

![TSNEResults](ReadmeImages/Screenshot%202023-07-03%20at%209.09.12%20AM.png)


## How to run:
The script was made to run locally on Apple M-series GPU/MPS, but will check and use CUDA if available. Will fall back to running on CPU (but it'll take forever). It's a significant speed-up running on MPS vs CPU. **To run on MPS, make sure your version of Pytorch supports this.** At the time of this writing, this was only available with the nightly version of Pytorch:

General pytorch install instructions:
https://pytorch.org/get-started/locally/

MacOS Pytorch Nightly install for MPS support:
```zsh
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

Script for MoCo can be run with:
```zsh
python3 main_moco.py
```
The script will start fine-tuning a ResNet-18 for 100-epochs using the MoCo-V2 framework.

## How to choose the dataset:

It's currently set up to download and train on the segmented cropped plants dataset found here: https://vision.eng.au.dk/plant-seedlings-dataset/

![CroppedPlantSeedlingDataset](ReadmeImages/Screenshot%202023-07-03%20at%209.03.10%20AM.png)

If you want to use your own dataset, make a folder called "data", and it'll read from there. 

The "data" folder should be structered like this:

data/
- class_x
  - xxx.tif
  - xxy.tif
  - ...
  - xxz.tif
- class_y
  - 123.tif
  - nsdf3.tif
  - ...
  - asd932_.tif

In the case the data is unlabeled, use:

data/
- dataset_name
  - xxx.tif
  - xxy.tif
  - ...
  - xxz.tif


---
---

## What is self-supervised learning?
Image classification is typical done using supervised algorithms—that is, algorithms that utilize a large dataset of images with corresponding ground-truth labels for each image. These supervised systems learn vector representations of the images’ features, that can then be used for downstream tasks such as classification or clustering into the different classes. The large cost of gathering and curating such large, labeled, datasets has spurred research in alternative ways to learn image feature representations. In the absence of labeled image data, this learning can be done using self-supervised deep-learning algorithms (“SSL”) that use the **unlabeled data itself as a supervisory signal** during training.

This is a good explaination using SimCLR as an example: https://amitness.com/2020/03/illustrated-simclr/

![SSLExplainer](ReadmeImages/Screenshot%202023-03-03%20at%204.42.09%20PM.png)
