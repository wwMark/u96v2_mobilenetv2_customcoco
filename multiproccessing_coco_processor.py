import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import cProfile
import PIL
# from PIL import ImageShow
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import math
from math import ceil, floor
import pathlib
from pathlib import Path
import os
import shutil
import multiprocessing as mp
from multiprocessing import Pool

def bbox_extractor(id, dataset):
  print("inside bbox_extractor")
  skip_count = id * process_size
  take_count = process_size
  if id == 19:
    take_count += process_size_remainder
  image_id = skip_count
  tile_dataset = dataset.skip(skip_count).take(take_count)
  for individual_data in tile_dataset.as_numpy_iterator():
    # objects is not sequence but a dict
    objects = individual_data['objects']
    # type of image here is np.ndarray
    image = individual_data['image']
    # convert ndarray to image
    image = Image.fromarray(image, 'RGB')
    image_width, image_height = image.size
    # bbox_list and label_list is ndarray with element of ndarray and np.int64 respectively
    bbox_list = objects['bbox']
    label_list = objects['label']
    for individual_bbox, individual_label in zip(bbox_list, label_list):
      y_min = floor(individual_bbox[0] * image_height)
      x_min = floor(individual_bbox[1] * image_width)
      y_max = ceil(individual_bbox[2] * image_height)
      x_max = ceil(individual_bbox[3] * image_width)
      label = interpret_result_to_label(label_mapping_list, individual_label)
      # copy the image content in the specified bbox coord. and save them in a directory
      save_path = project_root_path + 'multiprocessing_bbox_images\\coco_' + str(label) + '_' + str(image_id) + '_' + str(x_min) + '_' + str(y_max) + '_' + str(x_max) + '_' + str(y_min) + '.jpeg'
      image_id += 1
      cropped_image = image.crop((x_min, y_min, x_max, y_max))
      cropped_image = cropped_image.resize((224, 224))
      cropped_image.save(save_path, format='JPEG')

def interpret_result_to_label(label_mapping_list, label_id):  
  return label_mapping_list[label_id].replace('\n', '')

if __name__ == '__main__':
  project_root_path = 'C:\\Users\\Mark\\iCloudDrive\\hiwi\\刘博\\process_coco_and_train\\'

  # delete and remake bbox_images
  shutil.rmtree(project_root_path + 'multiprocessing_bbox_images')
  os.mkdir(project_root_path + 'multiprocessing_bbox_images')


  # download dataset and prepare label mapping file
  coco_train, info_train = tfds.load('coco/2014', split='train', shuffle_files=False, with_info=True)
  coco_test, info_test   = tfds.load('coco/2014', split='test', shuffle_files=False, with_info=True)
  label_mapping_file = open('coco2014_labels.txt', 'r')
  label_mapping_list = label_mapping_file.readlines()
  label_mapping_file.close()

  # iterate over the dataset, extract from coco the bounding box coordinates and extract new images according to the bbox coord. Then save to file system
  # count = 0
  # add multiprocessing code
  coco_cardinality = int(coco_train.cardinality())
  process_size = coco_cardinality / 20
  process_size_remainder = coco_cardinality % 20

  # multiprocessing code
  core_number = 20
  pool = mp.Pool(processes=core_number)
  print('working in multiprocessing code')
  for x in range(0, core_number):
    pool.apply(bbox_extractor, args=(x, coco_train,))
    print('in loop ', x)
  pool.close()
  pool.join()