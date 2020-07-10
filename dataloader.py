from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
import random
from config import *
import argparse
import time
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.utils import Sequence

import string
characters = string.digits + string.ascii_uppercase


class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for i in range(self.n_len)]
        for i in range(self.batch_size):
            random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y


data = CaptchaSequence(characters, batch_size=2, steps=100)
X, y = data[1]
print(X.shape, len(y), y[0].shape) # (2, 64, 128, 3)  (4, 2, 36)
# def decode(y):
#     y = np.argmax(np.array(y), axis=2)[:,0]
#     return ''.join([characters[x] for x in y])







#
# '''
# # 将字符串验证码转为在字母表中的位置(list), 方便转为ont-hot编码
# # print(tf.one_hot(index, depth=CAPTCHA_CLASSES, dtype=tf.int32))
# '''
# def encode(label):
#     index = [CAPTCHAES.find(i) for i in label]
#     return tf.one_hot(index, depth=CAPTCHA_CLASSES, dtype=tf.int32)
# '''
# 将位置转化为字符串验证码
# '''
# def decode(index):
#     label = "".join([CAPTCHAES[i] for i in index])
#     return label
# '''
# 将one_hot转为字符串验证码
# # shape=(32, 4, 36)
# '''
# def de_one_hot(one_hot):
#     batch = tf.argmax(one_hot, axis=-1) # shape=(32, 4)
#     if len(batch.shape) == 2:
#         res = []
#         for index in batch:
#             res.append(decode(index)) # index表示4位的索引
#     else: # 如果只有1张图片
#         res = decode(batch)
#     return res
#
# def load_dataset(root):
#     img_paths, labels = [], []
#     for name in sorted(os.listdir(root)):
#         img_paths.append(os.path.abspath(os.path.join(root, name)))
#         labels.append(encode(name[:CAPTCHA_LEN]))
#     return img_paths, labels
#
# def preprocess(x, y):
#     x  =tf.io.read_file(x)
#     x = tf.image.decode_jpeg(x, channels=3)
#     x = tf.cast(x, dtype=tf.float32) / 255.
#     y = tf.convert_to_tensor(y)
#     return x, y
#
# if __name__ == '__main__':
#     img_paths, labels = load_dataset("data/test")           # print(img_paths[1], labels[1])
#     db  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
#     db= db.shuffle(1000).map(preprocess).batch(32)   #  print(next(iter(db))[0].shape,next(iter(db))[1].shape)

    # print(next(iter(db))[0].shape, next(iter(db))[1].shape) # (32, 80, 170, 3) (32, 4, 36)
    # print(tf.transpose(next(iter(db))[1], perm=[1, 0, 2]).shape) # 标签必须换维度,要不然和网络的维度不一样
    # print(de_one_hot(next(iter(db))[1]))
















    # x = tf.io.read_file(img_paths[0])
    # x = tf.image.decode_png(x, channels=3)
    # plt.imshow(x)
    # plt.show()

    # x = Image.open(img_paths[0]).convert('RGB')
    # x.show()
    # print(len(img_paths), len(labels))


