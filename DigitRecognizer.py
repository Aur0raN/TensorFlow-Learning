import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


class callback(tf.keras.callbacks.Callback):  # Callback function to stop when accuracy is 99%
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") >= 0.99:
            print("\n Reached 99% Accuracy so canceling training")
            self.model.stop_training = True


mnist = tf.keras.datasets.mnist  # importing MNist dataset from keras API call

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255  # to get the values between 0 and 1


# plt.imshow(x_train[1])
# print(x_train[1])
# print(x_test[1])
# plt.show()


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # Images are given in 28 x 28
    tf.keras.layers.Dense(784, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

answer = model.fit(x_train,y_train,epochs=10,callbacks=[callback()])
