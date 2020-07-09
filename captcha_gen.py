from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
import random
from config import *
import argparse
import time
from PIL import Image
import os

def generator_captcha():
    width, height, n_len, n_class = WIDTH, HEIGHT, CAPTCHA_LEN, CAPTCHA_CLASSES
    chars =  CAPTCHAES
    gen = ImageCaptcha(width=width, height=height)
    str_ = ''.join([random.choice(chars) for _ in range(CAPTCHA_LEN)])
    image = Image.open(gen.generate(str_))
    return str_, image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--count', type=int, help="图片数量", default=10000) # 不填默认1万
    parser.add_argument('-m', '--mode', type=str, help="生成数据集的形式, train表示训练集，test为测试集,val为验证集", default='test')

    args = parser.parse_args()
    if args.mode == 'train':
        path = TRAIN_DIR
    elif args.mode == 'val':
        path = VAL_DIR
    elif args.mode == 'test':
        path = TEST_DIR
    else :
        raise Exception("请选择正确的模式")
    for i in range(args.count):
        now = str(int(time.time()))
        str_, image = generator_captcha()
        filename = str_ + "_" + now + ".png"
        image.save(os.path.join(path, filename))






# image.save(path + os.path.sep + filename)















    # group = parser.add_mutually_exclusive_group() # 互斥组

    # group.add_argument('--train_dir', type=str, help="训练集存放目录", default="./data/train")
    # group.add_argument('--val_dir', type=str, help="训练集存放目录", default="./data/val")
    # group.add_argument('--test_dir', type=str, help="训练集存放目录", default="./data/test")
    














# import string

# chars =  CAPTCHAES

# width, height, n_len, n_class = WIDTH, HEIGHT, CAPTCHA_LEN, CAPTCHA_CLASSES

# gen = ImageCaptcha(width=width, height=height)
# str_ = ''.join([random.choice(chars) for _ in range(4)])

# img = gen.generate_image(str_)

# plt.imshow(img)
# plt.show()

