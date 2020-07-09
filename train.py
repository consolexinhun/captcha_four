from config import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dataloader import encode, decode, load_dataset, preprocess
import time
from model import MyModel
from tqdm import tqdm


batch_size, lr, epochs = 32, 1e-3, 20
# train

img_paths, labels = load_dataset(TRAIN_DIR)
db_train  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
db_train = db_train.map(preprocess).shuffle(1000).batch(batch_size)


# val
img_paths, labels = load_dataset(VAL_DIR)
db_val  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
db_val = db_val.map(preprocess).shuffle(1000).batch(batch_size)

print("datasets loaded!")
# test
# img_paths, labels = load_dataset("data/test")
# db_test  = tf.data.Dataset.from_tensor_slices((img_paths, labels))
# db_test = db_test.shuffle(1000).map(preprocess).batch(batch_size)

model = MyModel()

optimizer = optimizers.Adam(lr=lr)
for epoch in tqdm(range(epochs)):
    for step, (x, y) in enumerate(db_train):
        label  = tf.transpose(y, perm=[1, 0, 2])
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(label, logits, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 10 == 0:
            print("epoch:{}, step:{}, loss:{}".format(epoch, step, float(loss)))

    total_correct = 0
    total_num = 0
    for x, y in db_val: # y(32, 4, 36)
        logits = model(x) # (4, 32, 36)
        out = tf.transpose(logits, perm=[1, 0, 2]) # (32, 4, 36)
        size = x.shape[0]

        correct = 0
        for i in range(size):
            arg_out = tf.argmax(out[i], axis=-1) # 4
            arg_y = tf.argmax(y[i], axis=-1) # 4

            flag = True
            for j in range(CAPTCHA_LEN):
                if not tf.equal(arg_y[j], arg_out[j]): # 4位验证码中如果有一个不相等就不算对
                    flag = False
                    break
            if flag:
                correct += 1

        total_correct  += correct
        total_num += size

    acc = total_correct / total_num
    print("epoch:{}, acc:{}".format(epoch, acc))

    # if epoch % SAVE_FREQ == 0:
    #     model.save("model/"+str(epoch)+".weight")


# model.load_weights()













































# callbacks = [EarlyStopping(patience=3), CSVLogger('cnn.csv'), ModelCheckpoint('cnn_best.h5', save_best_only=True)]
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(1e-3, amsgrad=True),
#               metrics=['accuracy'])
# model.fit_generator(db_train, epochs=100, validation_data=db_val, workers=4, use_multiprocessing=True,
#                     callbacks=callbacks)









