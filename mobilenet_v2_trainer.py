import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import cProfile
import PIL
from PIL import Image
import numpy as np
import pathlib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import math
from math import ceil, floor
import os
# from tensorflow_model_optimization.quantization.keras import vitis_quantize
# boolean to indicate whether to train model again
train_again = False

# check whether pretrained model already exist
project_root_path = pathlib.Path(__file__).parent.resolve()
pretrained_model_root_path = os.path.join(project_root_path, 'trained_model')
model_name = 'mobilenetv2'
pretrained_model_path = os.path.join(pretrained_model_root_path, model_name)

# function of interpreting label id to label name
def interpret_result_to_label(label_mapping_list, label_id):
  return label_mapping_list[label_id].replace('\n', '')

pretrained_model_is_empty = True if len(os.listdir(pretrained_model_path)) == 0 else False
if train_again or pretrained_model_is_empty:
  if train_again:
    print('Training model again...')
  else:
    print('No pretrained model {model} exists, now training new model...'.format(model=model_name))
  # set up GPU to avoid cublas error
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  print("Num GPUs Available: ", len(physical_devices))
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

  # download dataset and prepare label mapping list
  coco2014_cropped_train, info_train = tfds.load('coco2014_cropped_images', split='train', shuffle_files=True, with_info=True, as_supervised=True)
  coco2014_cropped_validation, info_validation = tfds.load('coco2014_cropped_images', split='validation', shuffle_files=True, with_info=True, as_supervised=True)
  # coco2014_cropped_test, info_test = tfds.load('coco2014_cropped_images', split='test', shuffle_files=True, with_info=True, as_supervised=True)
  label_mapping_list = info_train.features['label'].names

  '''
  # debug code to show dataset image
  single_image = coco2014_cropped.take(1)
  for example_image in single_image.as_numpy_iterator():
    converted_image = Image.fromarray(example_image['image'], 'RGB')
    converted_image.show()
  quit()
  '''

  # prepare model, use 1080: 720 resolution cause coco images 
  # model = tf.keras.applications.MobileNetV3Small(alpha=1.0, minimalistic=True, weights=None, include_top=True, classes=80, pooling=None, dropout_rate=0.2, classifier_activation='softmax')
  model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None, classes=80)

  # compile model
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

  # train model
  coco2014_cropped_train = coco2014_cropped_train.batch(32).prefetch(2)
  coco2014_cropped_validation = coco2014_cropped_validation.batch(32).prefetch(2)
  # coco2014_cropped_train = tf.keras.applications.mobilenet_v2.preprocess_input(coco2014_cropped_train)
  history = model.fit(x=coco2014_cropped_train, epochs=1, validation_data=coco2014_cropped_validation)
  model.save(pretrained_model_path)
  'Training finished.'
  print('Train history.')
  history.history
  'Model saved to {path}.'.format(path=pretrained_model_path)
else:
  'Pretrained {model} exists, no need to train from scratch.'
  load_model_path = pretrained_model_path
  model = tf.keras.models.load_model(load_model_path)
  label_mapping_file = open('coco2014_labels.txt', 'r')
  label_mapping_list = label_mapping_file.readlines()
  label_mapping_file.close()

'''
# evaluate model
'Evaluating model...'
result = model.evaluate(coco2014_cropped_test)
dict(zip(model.metrics_names, result))
'Evaluation finished.'
'''

# predict test image
test_image = Image.open(os.path.join(project_root_path, 'test_images', 'toothbrush.jpeg'))
test_image = test_image.resize((224, 224))
input = image.img_to_array(test_image)
input = np.expand_dims(input, axis=0)
# input = tf.keras.applications.mobilenet.preprocess_input(input)
prediction = model.predict(input)
print(prediction)

# interpret prediction
label_index = np.argmax(prediction)
print("label_index ", interpret_result_to_label(label_mapping_list, label_index))
quit()
'''
for element in cifar100_train_with_id.as_numpy_iterator():
  print(element['label'])
  if label_index == 0:
    print(element['id'])
    break
'''
# interpretation = next(cifar100_train.skip(label_index).take(1).as_numpy_iterator())
# print(interpretation)


'''
tf.executing_eagerly()

coco, info = tfds.load('coco', split='train', shuffle_files=False, with_info=True)
'''
'''
example = coco.take(1)
print('\nprinting example:')
print(example)
# tfds.show_examples(example, info)

for example_numpy in example.as_numpy_iterator():
  ex_image_array = example_numpy['image']
  ex_image = Image.fromarray(ex_image_array, 'RGB')
  # ImageShow.show(ex_image)
  ex_image.show()
'''

'''
small_train_dataset = coco.take(5)
for example in small_train_dataset.as_numpy_iterator():
  # get image
  image_numpy_array = example['image']
  # get label
  objects = example["objects"]
  label = objects["label"]
  print("label: ", label, "\n")
  constructed_image = Image.fromarray(image_numpy_array, 'RGB')
  constructed_image.show(title=label)
'''
'''
for image, label in coco.take(1):
  print(image)
  print(label)
'''
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
'''
'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
'''
'''
# rewrite model in Sequential API to Keras functional API
inputs = keras.Input(shape=(28, 28))
flatten = layers.Flatten()(inputs)
dense = layers.Dense(128, activation="relu")(flatten)
dropout = layers.Dropout(0.2)(dense)
outputs = layers.Dense(10, activation="relu")(dropout)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")


predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])

# model.save('train_results/model')

# model.save_weights('train_results/checkpoint/')

from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(model)

quantized_model = quantizer.quantize_model(calib_dataset=mnist)

quantized_model.save('./quantized_model')
'''


'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
image_girl = Image.open('girl.jpg')
image_girl = image_girl.resize((224, 224))
input = image.img_to_array(image_girl)
input = np.expand_dims(input, axis=0)
print("printing input shape ", input.shape)
input = tf.keras.applications.mobilenet.preprocess_input(input)
model = tf.keras.applications.mobilenet.MobileNet()
print(model.predict(input))
'''
'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# path = os.path.join(sys.path[0], 'girl.jpg')
image_girl = Image.open('girl.jpg')
# image_girl.show()
# resized_image_girl = image_girl.resize((224, 224))
# resized_image_girl.show()
# input = np.asarray(image_girl)
input = image.img_to_array(image_girl)
input = np.expand_dims(input, axis=0)
print("printing input shape ", input.shape)
input = tf.keras.applications.mobilenet.preprocess_input(input)
model = tf.keras.applications.MobileNetV3Large(alpha=1.0, minimalistic=False, include_top=True, weights='imagenet', classes=1000, pooling=None, dropout_rate=0.2, classifier_activation='softmax')
predictions = model.predict(input)
results = imagenet_utils.decode_predictions(predictions)
print(results)
'''
