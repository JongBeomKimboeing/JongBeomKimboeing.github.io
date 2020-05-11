---
layout: post
title: Tensorboard
description: "Tensorboard"
modified: 2020-05-10
tags: [김성훈,DL]
categories: [김성훈DL]
---
# tensorboard를 사용하여 그래프 그리기
```python
import tensorflow as tf
import os
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time #time import
import datetime #datetime import

rand.seed(777)
learning_rate = 0.01
batch_size = 100
training_epoch = 15
nb_classes = 10

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

x_train = tf.reshape(x_train, shape=[x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
x_test = tf.reshape(x_test, shape=[x_test.shape[0], x_test.shape[1]*x_test.shape[2]])

y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

start = time.time()

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=2000, input_dim = x_train.shape[1] , activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=2000, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=2000, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=2000, activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=nb_classes, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
tf.model.summary()
```
tensorboard를 위해 코드에 추가 시켜준다.
```python
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#경로를 병합하여 새 경로(파일) 생성

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq= 1)
# tf.keras.callbacks.TensorBoard-> log_dir= :log 디렉토리이름,  histogram_freq= 1 : 몇 epoch마다 histogram을 계산할지

history = tf.model.fit(x_train, y_train, batch_size= batch_size, epochs=training_epoch,callbacks=[tensorboard_callback])
#callbacks= : training동안 적용할 callback들의 list
```
```python
prediction = tf.model.predict(x_test)
print("prediction: \n",prediction)
accuracy = tf.model.evaluate(x_test, y_test)
print("accuracy: ",accuracy[1])

r = rand.randint(0, x_test.shape[0]-1)
print("Label:", np.argmax(y_test[r]))
print("predicted:", np.argmax(prediction[r]))
print("time: %d sec" %(time.time() - start))
plt.imshow(x_test[r].numpy().reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()
```
