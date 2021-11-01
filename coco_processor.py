import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import cProfile
import PIL
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

project_root_path = pathlib.Path(__file__).parent.resolve()
dataset_path = os.path.join(project_root_path, 'dataset')
coco_path = os.path.join(dataset_path, 'coco')
bbox_images_path = os.path.join(coco_path, 'bbox_images')

train_path = os.path.join(coco_path, 'train2014')
validation_path = os.path.join(coco_path, 'validation2014')
test_path = os.path.join(coco_path, 'test2014')

train_bbox_path = os.path.join(bbox_images_path, 'train2014')
validation_bbox_path = os.path.join(bbox_images_path, 'validation2014')
test_bbox_path = os.path.join(bbox_images_path, 'test2014')
Path(train_bbox_path).mkdir(parents=True, exist_ok=True)
Path(validation_bbox_path).mkdir(parents=True, exist_ok=True)
Path(test_bbox_path).mkdir(parents=True, exist_ok=True)

# function of interpreting label id to label name
def interpret_result_to_label(label_mapping_list, label_id):
    return label_mapping_list[label_id].replace('\n', '')

# check existence of directory of extracted bbox images
# train
'''
if Path(train_path).exists():
    'Directory {path} already exists, now deleting it...'.format(path=train_path)
    'Creating directory {path}...'.format(path=train_path)
    try:
        Path(train_path).mkdir(parents=True)
    except:
        'Error occured, failed to create {path}'.format(path=train_path)
    '{path} created.'
else:
    'Directory {path} does not exist, now creating...'
    try:
        Path(train_path).mkdir(parents=True)
    except:
        'Error occured, failed to create {path}'.format(path=train_path)
    '{path} created.'
# validation set
if Path(validation_path).exists():
    'Directory {path} already exists, now deleting it...'.format(path=validation_path)
    'Creating directory {path}...'.format(path=validation_path)
    try:
        Path(validation_path).mkdir(parents=True)
    except:
        'Error occured, failed to create {path}'.format(path=validation_path)
    '{path} created.'
else:
    'Directory {path} does not exist, now creating...'
    try:
        Path(validation_path).mkdir(parents=True)
    except:
        'Error occured, failed to create {path}'.format(path=validation_path)
    '{path} created.'
'''
'''
# test
if Path(test_path).exists():
  'Directory {path} already exists, now deleting it...'.format(path=test_path)
  'Creating directory {path}...'.format(path=test_path)
  try:
    Path(test_path).mkdir(parents=True)
  except:
    'Error occured, failed to create {path}'.format(path=test_path)
  '{path} created.'
else:
  'Directory {path} does not exist, now creating...'
  try:
    Path(test_path).mkdir(parents=True)
  except:
    'Error occured, failed to create {path}'.format(path=test_path)
  '{path} created.'
'''

# download dataset and prepare label mapping file
coco_train, info_train = tfds.load('coco/2014', split='train', data_dir=train_path, shuffle_files=False, with_info=True, download=True)
coco_validation, info_test = tfds.load('coco/2014', split='validation', data_dir=validation_path, shuffle_files=False, with_info=True, download=True)
coco_test, info_test = tfds.load('coco/2014', split='test', data_dir=test_path, shuffle_files=False, with_info=True, download=True)
label_mapping_file = open(os.path.join(project_root_path, 'coco2014_labels.txt'), 'r')
label_mapping_list = label_mapping_file.readlines()
label_mapping_file.close()

'''
# iterate over the train dataset, extract from coco the bounding box coordinates and extract new images according to the bbox coord. Then save to file system
'Extracting training set...'
# each image includes its variants has only one id
original_id = 0
# unique id for each individual extracted image
unique_id = 0
for individual_data in coco_train.as_numpy_iterator():
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
  print(original_id)
  for individual_bbox, individual_label in zip(bbox_list, label_list):
    y_min = floor(individual_bbox[0] * image_height)
    x_min = floor(individual_bbox[1] * image_width)
    y_max = ceil(individual_bbox[2] * image_height)
    x_max = ceil(individual_bbox[3] * image_width)
    label = interpret_result_to_label(label_mapping_list, individual_label)
    # copy the image content in the specified bbox coord. and save them in a directory
    save_path = train_path + '\\' + str(label) + '_' + str(original_id) + '_' + str(x_min) + '_' + str(y_max) + '_' + str(x_max) + '_' + str(y_min) + '_'
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize((224, 224))
    # rotation to augment dataset
    cropped_image.save(save_path + 'rotate0' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate90' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate180' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate270' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
  original_id += 1
'Extraction finished.'
'''
# each image includes its variants has only one id
original_id = 0
# unique id for each individual extracted image
unique_id = 0
current_splits = [coco_train, coco_validation, coco_test]
current_save_paths = [train_bbox_path, validation_bbox_path, test_bbox_path]
for current_split, current_save_path in zip(current_splits, current_save_paths):
    print("Current split is", current_split)
    print("Current save path is", current_save_path)
    for individual_data in current_split.as_numpy_iterator():
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
        if original_id % 1000 == 0:
            print("Finished processing", original_id, "images.")
        for individual_bbox, individual_label in zip(bbox_list, label_list):
            y_min = floor(individual_bbox[0] * image_height)
            x_min = floor(individual_bbox[1] * image_width)
            y_max = ceil(individual_bbox[2] * image_height)
            x_max = ceil(individual_bbox[3] * image_width)
            label = interpret_result_to_label(label_mapping_list, individual_label)
            # copy the image content in the specified bbox coord. and save them in a directory
            save_path = os.path.join(current_save_path , str(label) + '_' + str(original_id) + '_' + str(x_min) + '_' + str(
                y_max) + '_' + str(x_max) + '_' + str(y_min) + '_')
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image = cropped_image.resize((224, 224))
            # rotation to augment dataset
            cropped_image.save(save_path + 'rotate0' + '_' + str(unique_id) + '.jpeg', format='JPEG')
            unique_id += 1
            cropped_image = cropped_image.rotate(90)
            cropped_image.save(save_path + 'rotate90' + '_' + str(unique_id) + '.jpeg', format='JPEG')
            unique_id += 1
            cropped_image = cropped_image.rotate(90)
            cropped_image.save(save_path + 'rotate180' + '_' + str(unique_id) + '.jpeg', format='JPEG')
            unique_id += 1
            cropped_image = cropped_image.rotate(90)
            cropped_image.save(save_path + 'rotate270' + '_' + str(unique_id) + '.jpeg', format='JPEG')
            unique_id += 1
        original_id += 1

'''
# iterate over the test dataset, extract from coco the bounding box coordinates and extract new images according to the bbox coord. Then save to file system
'Extracting test set...'
# each image includes its variants has only one id
original_id = 0
# unique id for each individual extracted image
unique_id = 0
for individual_data in coco_test.as_numpy_iterator():
  print(individual_data)
  quit()
  # objects is not sequence but a dict
  objects = individual_data['objects']
  # type of image here is np.ndarray
  image = individual_data['image']
  # convert ndarray to image
  image = Image.fromarray(image, 'RGB')
  image_width, image_height = image.size
  # bbox_list and label_list is ndarray with element of ndarray and np.int64 respectively
  label = individual_data['label']
  print(label)
  # copy the image content in the specified bbox coord. and save them in a directory
  save_path = test_path + '\\' + str(label) + '_' + str(original_id) + '_' + '10' + '_' + '10' + '_' + '10' + '_' + '10' + '_'
  cropped_image = image.resize((224, 224))
  # rotation to augment dataset
  cropped_image.save(save_path + 'rotate0' + '_' + str(unique_id) + '.jpeg', format='JPEG')
  unique_id += 1
  cropped_image = cropped_image.rotate(90)
  cropped_image.save(save_path + 'rotate90' + '_' + str(unique_id) + '.jpeg', format='JPEG')
  unique_id += 1
  cropped_image = cropped_image.rotate(90)
  cropped_image.save(save_path + 'rotate180' + '_' + str(unique_id) + '.jpeg', format='JPEG')
  unique_id += 1
  cropped_image = cropped_image.rotate(90)
  cropped_image.save(save_path + 'rotate270' + '_' + str(unique_id) + '.jpeg', format='JPEG')
  unique_id += 1
  original_id += 1
'Extraction finished.'


# iterate over the test dataset, extract from coco the bounding box coordinates and extract new images according to the bbox coord. Then save to file system
'Extracting test set...'
# each image includes its variants has only one id
original_id = 0
# unique id for each individual extracted image
unique_id = 0
for individual_data in coco_test.as_numpy_iterator():
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
  print(original_id)
  print((bbox_list[0], label_list[0]))
  for individual_bbox, individual_label in zip(bbox_list, label_list):
    print((individual_bbox, individual_label))
    y_min = floor(individual_bbox[0] * image_height)
    x_min = floor(individual_bbox[1] * image_width)
    y_max = ceil(individual_bbox[2] * image_height)
    x_max = ceil(individual_bbox[3] * image_width)
    label = interpret_result_to_label(label_mapping_list, individual_label)
    # copy the image content in the specified bbox coord. and save them in a directory
    save_path = test_path + '\\' + str(label) + '_' + str(original_id) + '_' + str(x_min) + '_' + str(y_max) + '_' + str(x_max) + '_' + str(y_min) + '_'
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize((224, 224))
    # rotation to augment dataset
    cropped_image.save(save_path + 'rotate0' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate90' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate180' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
    cropped_image = cropped_image.rotate(90)
    cropped_image.save(save_path + 'rotate270' + '_' + str(unique_id) + '.jpeg', format='JPEG')
    unique_id += 1
  original_id += 1
'Extraction finished.'
'''
