# SpaceNet Buildings v1

This repository displays the work I did for SpaceNet [Buildings Footprint Extraction Challenge Round 1](https://spacenet.ai/spacenet-buildings-dataset-v1/). The challenge tasked competitors with finding automated methods for **extracting map-ready building footprints** from satellite imagery of the city **Rio de Janerio**. This helps to move toward more accurate, fully automated extraction of buildings aimed at creating better maps where they are most needed.

As a part of this project, I designed a **model training pipeline** using the Fastai Library.

## Taking a look at the Dataset

### Problems
The dataset is huge and has 6940 images, which is sufficient for a segmentation dataset. However, **the dataset is noisy**, with around 2500 images having no building labels at all. These are images of forest cover, entirely green with no buildings and do not contribute to training. These images make the training harder and more time consuming as the model has to go through more number of images per epoch. Along with this, the training loss and validation loss we get per epoch are also not correct. This is because these noisy images increase the number of images in the denominator while adding nothing to the loss in the numerator, giving a much lesser loss than the actual value. 

Moreover, another problem is that the **images don’t have good enough contrast**. A little more clarity could help distinguish between building rooftops, primarily when buildings are clustered together. 

The problem mentioned above persists in the labels as well. Whenever there’s a cluster of buildings together, the **labels are also clustered together**. This forces the model to train in a similar manner, i.e., clusters the predicted labels together when buildings are very close to each other. 

Another problem that persists in the labels is that some of **labels are incorrectly marked**. To give a building a shape of a proper rectangle, some buildings are assumed to extend into tress like the example shown below. This makes it hard for the model to train, thus giving less dice value. The labelling at the center can be seen as faulty on visual inspection. 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Faulty%20Labelling/Faulty%20Labelling.png)

## Data Augmentation
Firstly I removed noisy images from the dataset to make it easier to train and obtain correct metrics and loss values. 

Since the images removed are of forest cover, this information lost is recovered from the rest of the images which are included as there is sufficient forest cover in the included images for the model to learn that information.

Next, I moved to Data transforms apart from the default ones that fastai provides. 

    data = (src.transform(get_transforms(flip_vert=True, max_rotate=15, max_zoom=1.2, max_lighting=0.5,
                          xtra_tfms = [brightness(change=0.56, p=1), contrast(scale=(1, 1.75), p=1.)]), 
                          size = size, tfm_y=True).databunch(bs=bs).normalize(imagenet_stats))

The figure below shows the brightness and contrast adjustment transforms applied to the satellite images to make the distinction of boundaries between two buildings or buildings and background (roads, garden, trees, etc.) more apparent. 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Data%20Augmentated/DA%201.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Data%20Augmentated/DA%202.png)


## Model Pipeline

The model pipeline is built using the Fastai Library. The UNet model uses a **ResNet34 as the encoder** and a peculiar loss function which combines Cross-Entropy loss with Dice Loss. 

The model also uses some tweaks taught in the [fastai course](https://course.fast.ai/) like [one-cycle policy training](https://docs.fast.ai/callbacks.one_cycle.html) per epoch (`fit_one_cycle`), `self-attention=True`, `norm_type=NormType.Weight`, `.to_fp16()` (which helps increasing the batch size by reducing the size of floating point values) and data augmentation mentioned above. The [Final Project Notebook ](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Jupyter%20Notebooks/SpaceNet_Buildings_v1_Final_Project_Notebook.ipynb) walks the reader through the implementation details. 

### Loss Function

The loss function is a **weighed scheme** of `cross-entropy loss` and `dice loss` which provides a suitable dual loss function and is popular in Semantic Segmentation competitions. 

It is observed that cross-entropy loss **optimizes for pixel-level accuracy** whereas the dice loss helps in **improving the segmentation quality/metrics**. Weighted dice loss alone causes over-segmentation at the boundaries, whereas the weighted cross-entropy loss alone leads to very sharp contours with minor under-segmentation. More about this combined dual loss function can be read [in this paper.](https://arxiv.org/pdf/1801.05173.pdf)

### Self Attention
Convolutions in CNN’s primarily work with data that is localized, but this means it can ignore longer range dependencies within the image. The self-attention layer is designed to counter-act this and enforce attention to longer-range dependencies. 

In essence, **attention reweighs certain features of the network** according to some externally or internally (self-attention) supplied weights. This **reweighting of the channel-wise responses** in a certain layer of a CNN by using self-attention helps to model interdependencies between the channels of the convolutional features. 

In the original paper this is based on, they noted that this helps the network **focus more on object shapes rather than local regions** of fixed shapes. This property makes self-attention layers very important in semantic segmentation. 


## What works?
Different models were trained using the pipeline over various permutations of data augmentation and inclusion/exclusion of noisy images. 

The best results were found after removing the noisy images and augmenting the dataset by adjusting the contrast and brightness as specified in the data augmentation section above. The model with data augmentation converges to a better dice value faster than the rest combinations. 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Graphs/Dice%20Graphs.png)

Note that adding these tweaks makes the Dice value attain higher values, with lesser training. The loss function converges better and is less bumpy. 

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Graphs/Loss%20Graphs.png)

## How does it look like?

For simplicity, I have characterised the images in 4 types, i.e., Housing, Clustered Housing, Semi-Clustered Housing and Industrial Area. More pictures are uploaded in [this folder](https://github.com/the-catalyst/SpaceNet-Buildings-v1/tree/master/Result%20Pictures). 

### Housing
![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Housing/Housing%201.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Housing/Housing%202.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Housing/Housing%205.png)

### Clustered Housing

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Clustered/Clustered%201.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Clustered/Clustered%202.png)

### Semi-Clustered Housing
![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Semi-Clustered%20Housing/Semi-Clustered%20Housing%201.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Semi-Clustered%20Housing/Semi-Clustered%20Housing%202.png)

### Industrial Area
![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Industrial/Industrial%201.png)

![](https://github.com/the-catalyst/SpaceNet-Buildings-v1/blob/master/Result%20Pictures/Industrial/Industrial%202.png)
