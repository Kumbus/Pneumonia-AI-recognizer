import tensorflow as tf
import numpy as np
import os
import PIL
from PIL import Image
 
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pickle
import cv2
import pathlib
 
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
 
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
 
    return block
 
def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
 
    return block
 
def build_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(img_size, img_size, 3)),
 
      tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
      tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
      tf.keras.layers.MaxPool2D(),
 
      conv_block(32),
      conv_block(64),
 
      conv_block(128),
      tf.keras.layers.Dropout(0.2),
 
      conv_block(256),
      tf.keras.layers.Dropout(0.2),
 
      tf.keras.layers.Flatten(),
      dense_block(512, 0.7),
      dense_block(128, 0.5),
      dense_block(64, 0.3),
 
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])
 
  return model

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn
 
batch_size =  32
img_size = 180
data_dir = pathlib.Path('chest_xray/train')
test_dir = pathlib.Path('chest_xray/test')
#val_dir = pathlib.Path('chest_xray/val')
 
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split = 0.2,
  subset = "training",
  seed = 123,
  image_size = (img_size, img_size),
  batch_size = batch_size)
 
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split = 0.2,
  subset = "validation",
  seed = 123,
  image_size = (img_size, img_size),
  batch_size = batch_size)





class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)
 
 
count_normal = 1349
count_pneumonia = 3883
count_img = count_pneumonia + count_normal
weight_normal = (1 / count_normal) * count_img / 2.0
weight_pneumonia = (1 / count_pneumonia) * count_img / 2.0
class_weight = {0: weight_normal, 1: weight_pneumonia}
print(weight_normal)
print(weight_pneumonia)
print(class_weight)
 
with strategy.scope():
  model = build_model()
 
  METRICS = [
      'accuracy',
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
  ]
 
  model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=METRICS
  )
 
epochs=10
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

history = model.fit(
  train_ds,
  validation_data = val_ds,
  batch_size=batch_size,
  epochs = epochs,
  class_weight = class_weight,
  callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
  #validation_steps = (count_img * 0.2) // batch_size,
  #steps_per_epoch = (count_img * 0.8) // batch_size
)
print(model)
model.save('my_model.h5')

#model.predict()

