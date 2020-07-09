from config import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dataloader import encode, decode, load_dataset, preprocess

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        model = Sequential()
        # channel, repeat
        setting = [
            [64, 2],
            [128, 2],
            [192, 2],
            [192, 2],
            [192, 2]
        ]
        flag = True
        for channel, repeat in setting:
            for i in range(repeat):
                if flag:
                    model.add(layers.Conv2D(channel, input_shape=[HEIGHT, WIDTH, 3], kernel_size=3, padding="same",
                                            kernel_initializer='he_uniform'))
                    flag = False
                else:
                    model.add(layers.Conv2D(channel, kernel_size=3, padding="same", kernel_initializer='he_uniform'))
                model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
            model.add(layers.MaxPool2D(2))
        model.add(layers.Flatten())
        self.model = model
        self.dense = [layers.Dense(CAPTCHA_CLASSES, activation='softmax', name="Conv2D%d"%(i+1) ) for i in range(CAPTCHA_LEN)]

    def call(self, inputs, trainint=None):
        x = inputs
        x = self.model(x)
        out = []
        for j in self.dense:
            out.append(j(x))
        return out

if __name__ == '__main__':
    pass
    # x = tf.random.normal((32, 80, 170, 3))
    # net = MyModel()
    # y = net(x)
    # print(np.array(y).shape) # (4, 32, 36)






# l = [layers.Dense(CAPTCHA_CLASSES, activation='softmax', name="Conv2D%d"%(i+1) ) for i in range(CAPTCHA_LEN)]

# net = Sequential([
#     [layers.Dense(CAPTCHA_CLASSES, activation='softmax', name="Conv2D%d"%(i+1) ) for i in range(CAPTCHA_LEN)]
# ])
# tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, dpi=200)
