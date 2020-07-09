from config import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dataloader import encode, decode, load_dataset, de_one_hot, preprocess
import time
from model import MyModel
from tqdm import tqdm
from matplotlib import pyplot as plt

# model = MyModel()
# model.load_weights("model/")

# test
img_paths, labels = load_dataset(TEST_DIR)
db_test  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
db_test = db_test.map(preprocess).shuffle(1000).batch(32) #
sample = next(iter(db_test))
x, y = sample[0], sample[1]
index = 2
plt.imshow(x[index])
plt.axis('off')
plt.show()
true_label  = de_one_hot(y[index])
print("True Label",true_label)












# for x, y in db_test:  # y(32, 4, 36)
#     logits = model(x)  # (4, 32, 36)
#     out = tf.transpose(logits, perm=[1, 0, 2])  # (32, 4, 36)
#     size = x.shape[0]
#
#     correct = 0
#     for i in range(size):
#         arg_out = tf.argmax(out[i], axis=-1)  # 4
#         arg_y = tf.argmax(y[i], axis=-1)  # 4
#
#         flag = True
#         for j in range(CAPTCHA_LEN):
#             if not tf.equal(arg_y[j], arg_out[j]):  # 4位验证码中如果有一个不相等就不算对
#                 flag = False
#                 break
#         if flag:
#             correct += 1
#
#     total_correct += correct
#     total_num += size
#
# acc = total_correct / total_num
# print("epoch:{}, acc:{}".format(epoch, acc))