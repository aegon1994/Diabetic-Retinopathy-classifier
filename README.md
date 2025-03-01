# Diabetic-Retinopathy-classifier
This is a classifier which could classify eyeballs which are in Diabetic-Retinopathy into three different types. This project refer to the competition in [Diabetic Retinopathy Classification](https://www.kaggle.com/competitions/retinopathy-classification-sai/team).
<img src="https://github.com/aegon1994/Diabetic-Retinopathy-classifier/blob/main/image/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-03-01%20154415.png?raw=true">

I will show an efficienNet model which could extract the features from high resolution medical images and classified medical images by those features.

## preparing stage
In this project, I built this process and train this model in PyTorch framework in local environment.
If you want to rebuilt this project or implemented this model, I have recommended you to build a Pytorch framework in your computer. The guide of Pytorch is here.
[How to start the Pytorch](https://pytorch.org/get-started/locally/).

### dataset and labels
The dataset was provided by [Eric Li](https://www.kaggle.com/taipingeric), the host of this competition.
Those datas is the medical images of eyeballs in Diabetic-Retinopathy and there are 2526 imnages in training dataset, and 522 images in testing data. There are three types in training datas, They are 0:1215, 1:560, 2:311.

This project was developed in Pytorch, so I didn't need to built labels of datas. In Pytorch, ImageFolder would produce labels according to the order of Sub-folders in the folder which is the location of your datas when it loaded the dataset.

Luckily, our type names of dataset is the numbers which order is from 0 to 2 and our datas is not too many, it means I could classify training datas manually and produce labels by ImageFolder.
### model
I implemented this project by efficienNet, I will introduce this model below.

### Computer hardware
My CPU is intel-i7-11th, GPU is NVIDIA GeForce RTX 3060, Size of memory is 40GB.

## model 
In this time, I implemented this project by efficienNet. EfficienNet is a high efficiency CNN model. Here is the architecture of EfficienNet<img src="https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png">
