---
layout: post
title: Initialization
description: "Initialization"
modified: 2020-05-10
tags: [김성훈,DL]
categories: [김성훈DL]
---
# xavier initialization을 추가한 NN
accuracy가 96%
```python
import tensorflow as tf
import os
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time
import datetime

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
#-----------------------------------------------------------------------------------------------------------------------------------
tf.model.add(tf.keras.layers.Dense(units=1000, input_dim = x_train.shape[1] , kernel_initializer= 'glorot_normal',activation='relu'))
# kernel_initializer= 'glorot_normal' : xavier initialization을 이용하여 weight(kernel) 값들을 초기화 시킨다.
# 모든 weight들을 xavier initialization을 이용해야 하기 때문에 모든 layers에 대해 초기화 시켜야한다.
# hidden layer의 각 층의 input과 output을 1000로 지정한다.
# hidden layer의 각 층의 input과 output을 2000으로 지정하면 오히려 accuracy가 떨어지는데, 이는 over fitting에 의해 일어나는 현상이다.
#-----------------------------------------------------------------------------------------------------------------------------------
tf.model.add(tf.keras.layers.Dense(units=1000, kernel_initializer= 'glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=1000, kernel_initializer= 'glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=1000,kernel_initializer= 'glorot_normal', activation='relu'))
tf.model.add(tf.keras.layers.Dense(units=nb_classes,kernel_initializer= 'glorot_normal', activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=['accuracy'])
tf.model.summary()

log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#경로를 병합하여 새 경로(파일) 생성
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq= 1)
# tf.keras.callbacks.TensorBoard-> log_dir= :log 디렉토리이름,  histogram_freq= 1 : 몇 epoch마다 histogram을 계산할지

history = tf.model.fit(x_train, y_train, batch_size= batch_size, epochs=training_epoch,callbacks=[tensorboard_callback])
#callbacks= : training동안 적용할 callback들의 list
prediction = tf.model.predict(x_test)
print("prediction: \n",prediction)
evaluation = tf.model.evaluate(x_test, y_test)
print("loss: ",evaluation[0])
print("accuracy: ",evaluation[1])

y_predicted = tf.model.predict(x_test)
for i in range(0, 10):
    random_index = rand.randint(0, x_test.shape[0]-1)
    print("index: ",random_index,"actual y: ",np.argmax(y_test[random_index]), "predicted y: ",np.argmax(y_predicted[random_index]))

r = rand.randint(0, x_test.shape[0]-1)
print("Label:", np.argmax(y_test[r]))
print("predicted:", np.argmax(prediction[r]))
print("time: %d sec" %(time.time() - start))
plt.imshow(x_test[r].numpy().reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()
```
# 고찰
initialization을 잘 해주면 accuracy가 더 올라간다.<br>
initialization에는 대표적으로<br>
Xavier initialization<br>
He initialization<br>
이 있다.
