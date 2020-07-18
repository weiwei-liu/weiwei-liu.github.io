---
layout: post
title: Real Time Cherry Detection
subtitle: Application of deep learning for sweet cherry phenotyping
cover-img: /assets/img/cherry.jpg
thumbnail-img: /assets/img/cherry_thumb.jpg
tags: [OpenCV, phenotyping,YOLO,CNN, transfer learning]
---

New cultivars of fruits are known to significantly increase the economic growth and success of the horticulture industry. The process of tree fruit breeding and germplasm development involves development crosses of distinct genotypes, followed by several stages of evaluation of the germplasm for numerous traits repeatedly over multiple years and locations before the commercialization of a new cultivar.

Traditionally, in each phase of fruit germplasm evaluation, the selection and evaluation of fruit traits from each genotype of germplasm are carried out manually, which involves a lot of tedious and repetitive tasks such as fruit counting, size measurement, and color classification. The tree fruit breeding process is therefore vulnerable to workers fatigue and subjectivity, resulting in errors and inconsistencies in the fruit traits data collected.

With recent advancements in image analysis, computer vision and artificial intelligence, many of the aforementioned manual tasks involved in the process of tree fruit breeding can be fully or partially automated. In this project, we applied the state of art object detection technique, [YOLO](https://github.com/AlexeyAB/darknet)(you only look once), to train a Convolutional Neural network model, and combined with [OpenCV](https://github.com/opencv/opencv) and python packages to build a real time cherry detection application.

## Data labelling

To train a Yolo model that is customized to our cherry detection usage, we manually labelled 100 cherry images using [LabelImg](https://github.com/tzutalin/labelImg).

![labelled image](/assets/img/label.png)

While LabelImg is easy to use, it is important to bear in mind that the labeling quality will affect the accuracy of the trained model. So, it is crucial to make sure the bounding box is drew accurately and each object in the image is properly labelled.

## Model Training

There are several different object detection algorithms using CNN(convolutional neural network) methods. Basically, these techniques fall into two categories. One is based on feature proposition and then images classification. They are implemented in two stages. First, they propose regions of interest in an image. Second, they classify these regions using CNN. A widely known example of this type of algorithm is the Region-based convolutional neural network (RCNN) and its cousins Fast-RCNN, Faster-RCNN. Another group of algorithms is implemented in one stage. Instead of selecting interesting regions of an image, they predict classes and bounding boxes for the whole image in one run of the algorithm, aka you only look once. The best known examples are Yolo and SSD (Single Shot Multibox Detector).

In this project, we used darknet, which is an open source neural network framework that implements Yolo with C and CUDA technology incorporating GPU computation. Yolo can be implemented with different image classification algorithms, such as imagenet, darknet, resnet, efficientnet, etc. In this project, we used [efficientnet](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html) which introduced a compound scaling method to scale up the network like resnet and mobilenet.    

![efficient net architecture](/assets/img/enet_architecture.png)

### Transfer Learning

Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In our case, the original Yolo-efficientnet weights are obtained by training the model on coco image data. We trained our model based on these transfered weights with only 50 images.

So, after labelling images with bounding boxes, A Yolo-efficientnet model was trained on google colab GPU. The model, which trained on only 50 images, achieved over 99% average precision and 92% average IOU (intersection over union).

<div style="text-align:center">
<img src="/assets/img/cherry_result.jpg" alt="cherry detection results" width="600" height="500"/>
</div>

## Real time cherry detection

After making sure the model had achieved satisfying results, we build a real time cherry detection application using OpenCV and other python packages. Besides, the application can also store the predicted results by our model, and then automatically analyze the bounding boxes sizes and extract colour information.

<div style="text-align:center">
<img src="/assets/img/cherry.gif" alt="realtime detection results" width="600" height="400"/>
</div>
