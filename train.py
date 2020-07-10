from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
from dataloader import CaptchaSequence
import string
characters = string.digits + string.ascii_uppercase
data = CaptchaSequence(characters, batch_size=2, steps=1000)

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *

train_data = CaptchaSequence(characters, batch_size=128, steps=1000)
valid_data = CaptchaSequence(characters, batch_size=128, steps=100)
callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]

from model import model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-3, amsgrad=True),
              metrics=['accuracy'])
model.fit_generator(train_data, epochs=100, validation_data=valid_data, workers=4, use_multiprocessing=True,
                    callbacks=callbacks)


