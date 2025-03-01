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
Those datas is the medical images of eyeballs in Diabetic-Retinopathy and there are 2526 images in training dataset, and 522 images in testing data. There are three types in training datas, They are 0:1215, 1:560, 2:311.

This project was developed in Pytorch, so I didn't need to built labels of datas. In Pytorch, ImageFolder would produce labels according to the order of Sub-folders in the folder which is the location of your datas when it loaded the dataset.

Luckily, our type names of dataset is the numbers which order is from 0 to 2 and our datas is not too many, it means I could classify training datas manually and produce labels by ImageFolder.
### model
I implemented this project by efficienNet, I will introduce this model below.

### Computer hardware
My CPU is intel-i7-11th, GPU is NVIDIA GeForce RTX 3060, Size of memory is 40GB.

## model 
In this time, I implemented this project by efficienNet. EfficienNet is a high efficiency CNN model. Here is the architecture of EfficienNet<img src="https://1.bp.blogspot.com/-DjZT_TLYZok/XO3BYqpxCJI/AAAAAAAAEKM/BvV53klXaTUuQHCkOXZZGywRMdU9v9T_wCLcBGAs/s1600/image2.png">

The feature of efficienNet is it will find the basic CNN network architecture with Neural Architecture Search (NAS) and set Depth, Width and resolution by Compound Scaling at same time.
That is why EfficientNet outperforms other CNN architectures by achieving higher accuracy while requiring relatively fewer computational resources.

In the graph, we could see the main stem of the architecture of EfficienNet is different size MBConv layers. MBConv including depthwise separable convolution and an SE (Squeeze-and-Excitation) block could help to extract features in key area. Depthwise separable convolution in MBConv could capture local detail features, it is more efficient than other CNNs in computational environment.

SE block is a sophisticated mechanism designed to enhance the network’s ability to focus on the most informative features in the feature maps. This process significantly improves the representational power of the model

In this project, the reason I chose this model to classify our data are:

1. I implemented this project on my laptop, My computational resources is limited. EfficienNet could achieve better performance in limited computational environment.

2. My dataset is medical images, it is high resolution images. I need high feature extracting power for this project. Compound Scaling could extract detail features in high resolution images.
 
3. This dataset is unbalanced dataset, the performance is better with EfficientNet in unbalanced dataset.

## Process

classified images by types manually -> feature engineering one, enhanced the saturation in HSV of specific color(in this project, is red and brown) ->  feature engineering two, enhanced contrast ratio of images -> computed weights of unbalanced dataset for computation of loss -> seted loss function(for unbalanced dataset, I calculate loss with focalloss) -> calculated mean and std of dataset for data loading -> data loading ->seted class for features learning enhancing(in this project, I chose mixup and cutmix) -> trained efficientnet model with k-fold -> saved weight of model and predicted the result.

### data labeling
In this project, there are three types called 0,1,2. I introduced Pytorch could label datas auto by address when it loaded datas. The only thing I needed to do was classify training data to correct subfolder. I classified data manually because data is not too many.

### feature engineering


   
