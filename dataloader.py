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
'''
# 将字符串验证码转为在字母表中的位置(list), 方便转为ont-hot编码
# print(tf.one_hot(index, depth=CAPTCHA_CLASSES, dtype=tf.int32))
'''
def encode(str_):
    index = [CAPTCHAES.find(i) for i in str_]
    one_hot = tf.one_hot(index, depth=CAPTCHA_CLASSES, dtype=tf.float32)
    one_hot = tf.reshape(one_hot, (-1,)) # 变为向量 # print(one_hot)
    return one_hot
'''
将位置转化为字符串验证码
'''
def decode(one_hot):
    index = tf.reshape(one_hot, (CAPTCHA_LEN, CAPTCHA_CLASSES))
    str_ = "".join([CAPTCHAES[tf.argmax(i)] for i in index]) # print(str_)
    return str_

'''
加载数据集
root : 数据集目录
'''
def load_dataset(root):
    img_paths, labels = [], []
    for name in sorted(os.listdir(root)):
        img_paths.append(os.path.abspath(os.path.join(root, name)))
        labels.append(encode(name[:CAPTCHA_LEN]))
    return img_paths, labels

def preprocess(x, y):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=1)
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y)
    return x, y

if __name__ == '__main__':
    # one = encode("BK7H")
    # de = decode(one)

    img_paths, labels = load_dataset("data/test")

    db  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    db= db.map(preprocess).shuffle(1000).batch(32)
    (x, y) = next(iter(db))
    plt.imshow(tf.squeeze(x[0]), cmap='gray')
    plt.show()
    print(decode(y[0]))


    # for i, (x, y) in enumerate(db):
    #     print(i)
    #     print(x.shape, y.shape)


























    # x = tf.io.read_file(img_paths[0])
    # x = tf.image.decode_png(x, channels=3)
    # plt.imshow(x)
    # plt.show()

    # x = Image.open(img_paths[0]).convert('RGB')
    # x.show()
    # print(len(img_paths), len(labels))


