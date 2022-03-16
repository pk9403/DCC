# Train face recognition tasks with LFW datasets

import tensorflow as tf
import numpy as np
import sklearn
import cv2

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from dcc import transform
from model import create_model

HIGHT = 218
WIDTH = 178
CHANNEL = 1
a, b, r = 4, 4, 64
EPOCHS = 50

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Load Data
lfw_dataset = sklearn.datasets.fetch_lfw_people(min_faces_per_person=100)
n_samples, h, w = lfw_dataset.images.shape
y = lfw_dataset.target
target_names = lfw_dataset.target_names
n_classes = target_names.shape[0]

# Preprocessing
X = np.zeros((lfw_dataset.images.shape[0], HIGHT, WIDTH, CHANNEL))

for k in range(lfw_dataset.images.shape[0]):
    X[k,:,:,0] = cv2.resize(lfw_dataset.images[k], dsize=(WIDTH, HIGHT))

X = X/255.0

# Split data to train and test set
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
  #tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
  #tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

for k in range(x_train.shape[0]):
    if(y_train[k]!=2):
        augmented_image = data_augmentation(x_train[k].reshape(1, HIGHT, WIDTH, CHANNEL))
        x_train = np.append(x_train, augmented_image, axis=0)
        y_train = np.append(y_train, y_train[k])

# DCC to training data  
x_train_dct = np.zeros(x_train.shape)
for k in range(x_train.shape[0]):
    x_train_dct[k] = transform(x_train[k], a, b, r)

# Create model
model = create_model(input_shape=(HIGHT, WIDTH, CHANNEL), num_class=n_classes)

# Train
with tf.device('/GPU:0'):
    hist = model.fit(x_train_dct, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

# Test
y_pred = model.predict(x_test)
final_pred = []

for k in range(len(y_pred)):
    final_pred.append(np.argmax(y_pred[k], axis=-1))

print(precision_recall_fscore_support(y_test, final_pred, average='weighted'))