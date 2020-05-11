---
layout: post
title: MNIST with Deep CNN
description: "MNIST with Deep CNN"
modified: 2020-05-10
tags: [김성훈,DL]
categories: [김성훈DL]
---
# Deep CNN (layer, initializer, regularization 추가)
```python
import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 225
x_test = x_test / 225
print(x_train.shape) #(60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0] ,28 ,28 ,1) # (60000, 28, 28, 1) #6000개의 데이터를 28*28의 이미지로 변환하고, 색은 하나이다.
print(x_train.shape)
print(x_test.shape) #(10000, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

learning_rate = 0.001
training_epoch = 12
batch_size = 128
drop_out_rate = 0.5

#layer1
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
# tf.keras.layers.Conv2D-> filters= : fiter의 개수, kernel_size=: kernal 크기,  input_shape=: input data의 모양
#                       -> strides=(1, 1): filter를 얼마만큼의 stride로 움직일 것인가.
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer2
tf.model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3), activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer3
tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding="same",activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer3(Fully connected)
tf.model.add(tf.keras.layers.Flatten())
tf.model.add(tf.keras.layers.Dense(units=200, kernel_initializer='glorot_normal',activation='relu'))
#layer4(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=200, kernel_initializer='glorot_normal',activation='relu'))
#layer5(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=100, kernel_initializer='glorot_normal',activation='relu'))
#layer(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=10, kernel_initializer='glorot_normal', activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()

tf.model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epoch)

y_predicted = tf.model.predict(x_test)
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(y_predicted[random_index]))

evaluation = tf.model.evaluate(x_test, y_test)
print('loss: ', evaluation[0])
print('accuracy', evaluation[1])
```
# 결과
## training set의 loss와 accuracy
```python
loss: 0.0400 - accuracy: 0.9904
```
## 10개의 random한 test set에 대한 label과 prediction 비교
```python
index:  5070 actual y:  9 predicted y:  9
index:  1986 actual y:  1 predicted y:  1
index:  3125 actual y:  7 predicted y:  7
index:  2285 actual y:  7 predicted y:  7
index:  9027 actual y:  3 predicted y:  3
index:  6230 actual y:  9 predicted y:  9
index:  514 actual y:  6 predicted y:  6
index:  1395 actual y:  2 predicted y:  2
index:  1784 actual y:  7 predicted y:  7
index:  1642 actual y:  2 predicted y:  2
```
## test set의 loss와 accuracy
```python
loss:  0.04643920457083477
accuracy 0.9883
```
# 고찰
CNN layer를 3층, FC layer를 3층으로 구성하여 모델을 만들었다.<br>
여기서는 Xavier initializer와 Dropout regularization을 추가시켰다.<br>
그 결과 accuracy의 향상과 overfitting 감소를 관찰할 수 있었다.<br>
다음 실습에서는 Ensemble학습을 도입하여 accuracy를 더욱 높혀볼 예정이다.<br>
(참고로, ensemble학습은 김성훈 강좌에 tf2로 변환한 코드가 없어 직접 변환했다.)
