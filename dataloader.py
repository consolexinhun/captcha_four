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
def encode(label):
    index = [CAPTCHAES.find(i) for i in label]
    return tf.one_hot(index, depth=CAPTCHA_CLASSES, dtype=tf.int32)
'''
将位置转化为字符串验证码
'''
def decode(index):
    label = "".join([CAPTCHAES[i] for i in index])
    return label
'''
将one_hot转为字符串验证码
# shape=(32, 4, 36)
'''
def de_one_hot(one_hot):
    batch = tf.argmax(one_hot, axis=-1) # shape=(32, 4)
    if len(batch.shape) == 2:
        res = []
        for index in batch:
            res.append(decode(index)) # index表示4位的索引
    else: # 如果只有1张图片
        res = decode(batch)
    return res

def load_dataset(root):
    img_paths, labels = [], []
    for name in sorted(os.listdir(root)):
        img_paths.append(os.path.abspath(os.path.join(root, name)))
        labels.append(encode(name[:CAPTCHA_LEN]))
    return img_paths, labels

def preprocess(x, y):
    x  =tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.convert_to_tensor(y)
    return x, y

if __name__ == '__main__':
    img_paths, labels = load_dataset("data/test")           # print(img_paths[1], labels[1])
    db  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    db= db.shuffle(1000).map(preprocess).batch(32)   #  print(next(iter(db))[0].shape,next(iter(db))[1].shape)

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


