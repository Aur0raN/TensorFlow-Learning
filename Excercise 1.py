import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow import keras

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


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)  # Your Code Here#
    ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)  # Your Code Here#
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')  # Your Code Here#)
    model.fit(xs, ys, epochs=500)  # Your Code here#
    return model.predict(y_new)[0]


prediction = house_model([7.0])
print(prediction)
