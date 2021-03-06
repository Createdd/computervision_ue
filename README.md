# Unsupervised person localization in wilderness search and rescue

# Table of contents

  - [Introduction](#introduction)
  - [Challenge](#challenge)
    - [Data extraction](#data-extraction)
  - [Methods and solution](#methods-and-solution)
    - [0. Image pre-processing](#0-image-pre-processing)
    - [1. Color channels approach](#1-color-channels-approach)
    - [2. Autoencoder approach](#2-autoencoder-approach)
      - [Initial idea](#initial-idea)
      - [Model Architecture](#model-architecture)
      - [Implementation and findings](#implementation-and-findings)

## Introduction

This is part of the exercise class "UE Computer Vision, Oliver Bimber / Indrajit Kurmi, 2021W" at the JKU Austria.
- The responsible institute at JKU is https://www.jku.at/en/institute-of-computer-graphics/
- Whereas they have a specific research project
for [Search and rescue with airborne optical sectioning](https://www.nature.com/articles/s42256-020-00261-3.epdf?sharing_token=CkVF30c-ohDFg7Bfz7vbXNRgN0jAjWel9jnR3ZoTv0Njw2M16sXA0c1i0-K0I8hyWAyPHw0VoEqSzrkBwYYyW6fhTSE6UR1hLVXodIJxrUXLGCuefrcgODgq7zmQeEDTqcs5bDAPpwteMKEXPcztPtUexI1JTEkxXmS4opWo-LA%3D)
- For further projects see: https://www.jku.at/en/institute-of-computer-graphics/research/projects/2021

In this lab project, we had to implement an unsupervised person localization algorithm.

## Challenge

![challenge.png](static/challenge.png)

### Data extraction

![data_method.png](static/data_method.png)

## Methods and solution

### 0. Image pre-processing

![img.png](static/processing1.png)

![img_1.png](static/processing2.png)

### 1. Color channels approach

#### Method
![img_2.png](static/colorchannels1.png)

![img_3.png](static/colorchannels2.png)

![img.png](static/pipeline.png)

![img_1.png](static/results.png)

- OpenCV functions cv2.findContours() and cv2.boundingRect()
- Pad by 2 pixels in each direction
- Merge overlapping bounding boxes
- Pad by 7 pixels in x and 4 pixels in y
- Choose the biggest bounding box of blue and red image, respectively
- Remove if x<24 or y<18
- If no detections -> lower threshold for binary image and start from 1
- Merge detections overlapping between the two images
- Pad detections to a minimum size of 38x30 px


#### Advantages and disadvantages
- Advantage:
    - can distinguish people from other objects by detecting movement
- Disadvantages:
    - bias towards detecting people wearing blue or red ??? problems finding people with green clothing
    - cannot detect people that are not moving or moving too little
    
### 2. Autoencoder approach

The autoeconder is implemented
in [anomaly_detection_autoencoder_SAR_JKU.ipynb](anomaly_detection_autoencoder_SAR_JKU.ipynb)

#### Initial idea

Going through various research papers on anomaly detection, we decided to try out an Autoencoder approach for this task

- Autoencoder -> encoder-decoder system to reconstruct the input as the output.
- Train a convolutional autoencoder so that it will reconstruct an image from the normal data with a smaller
  reconstruction error, but reconstruct an image from the anomaly data with a larger reconstruction error
- Our solution decides if an image is from the normal data or from the anomaly data based on a threshold of the
  reconstruction error.

- the model is encouraged to learn to precisely reproduce the most frequently observed characteristics
- when facing anomalies, the model should worsen its reconstruction performance.
- after training, the autoencoder will accurately reconstruct normal data, while failing to do so with unfamiliar
  anomalous data
- reconstruction error (the error between the original data and its low dimensional reconstruction) is used as an
  anomaly score to detect anomalies
- we are aware that autoencoding models can, be very good at reconstructing anomalous examples and consequently not able
  to reliably perform anomaly detection

#### Model Architecture

![img_4.png](static/model_architecture.png)

- Base is a Convolutional autoencoder for image denoising from official Keras docs
- Adapted loss for Structural Similarity Index (SSIM)
- Decided for that because
  - Relatively straight forward to tune
  - Simple architecture
  - Sufficient for our image detection problem

#### Implementation and findings

Over the course of the implementation it became apparent, that

- properly pre-processed images improve the performance of the autoencoder a lot
- a deep convolutional autoencoder is sufficient to reproduce the images properly
- the autoencoder should be trained with color images as the color provides most of the information for the task
- the biggest challenge is the length of training as
    - too short training shows too many reconstruction errors
    - too long training reconstructs anomalies
- as well as the threshold for finding the most useful SSIM differences

![img_5.png](static/reconstruction.png)
Reconstruction worked well

![img_6.png](static/activation_layers.png)

Visualization of activation layers over RBG channels, showing stronger activations for red and blue channel

![img_8.png](static/activated_anomaly.png)

Indication of finding the anomalies as desired.

![img_9.png](static/threshold_anomaly.png)

Finding the proper threshold for SSIM differences

![img.png](static/locating_human.png)

---


The project was implemented over the course of a semester at university.

In the end we implemented the whole pipeline to fit the corresponding grading criteria.


