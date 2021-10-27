## Overview

This repository contains python code for extracting images from images of coco dataset, preprocessing them, constructing
tensorflow custom dataset from them and training a mobilnetv2 model with them.

## Usage and Project structure
Execution order should be ```coco_processor.py``` &#8594; ```mobilenet_v2_train.py```.

The custom coco dataset is downloaded and preprocessed by ```coco_processor.py```. This code will download the coco
dataset into the ```dataset``` directory.

The root directory contains ```mobilenet_v2_train.py``` file for training mobilenetv2 model. It will save the trained
model inside the ```trained_model``` directory.

The directory ```coco2014_cropped_images``` contains all the files of a custom tensorflow dataset. To change data
example retrieving behaviour of the dataset or dataset path, please edit the ```coco2014_cropped_images.py``` file.

