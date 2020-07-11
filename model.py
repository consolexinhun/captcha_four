from config import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


model = tf.keras.Sequential([
    layers.Conv2D(32, input_shape=[HEIGHT, WIDTH, 1], kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Activation('relu'),
    layers.Conv2D(32, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),


    layers.Conv2D(64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Activation('relu'),
    layers.Conv2D(64, kernel_size=3, padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Activation('relu'),
    layers.MaxPooling2D(2),

    layers.Flatten(),

    layers.Dense(1024),
    layers.Dropout(0.2),
    layers.Activation('relu'),

    layers.Dense(CAPTCHA_LEN*CAPTCHA_CLASSES)
])

# tf.keras.utils.plot_model(model, to_file="cnn.png", show_shapes=True, dpi=100)
