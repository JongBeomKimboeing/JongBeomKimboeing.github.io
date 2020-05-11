---
layout: post
title: XOR with wide and deep NN
description: "XOR with wide and deep NN"
modified: 2020-05-08
tags: [김성훈,DL]
categories: [김성훈DL]
---
# wide하고 deep한 NN을 이용한 XOR문제 해결 
```python
import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=2, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
tf.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#optimizer를 SGD로 할 경우 vanishing gradient에 의해서 학습이 잘 안된다.
#그러므로  vanishing gradient 문제를 해결하기 위해 Adam을 이용하거나 relu를 이용한다.
tf.model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.1),metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_data,y_data, epochs=5000)

prediction = tf.model.predict(x_data)
print("prediction: \n",prediction)
```
예측결과<br>
-> 예측 결과가 더 극대화 돼서 나타남.
```python
prediction: 
 [[2.2947788e-06]
 [9.9999905e-01]
 [9.9999905e-01]
 [2.8610229e-06]]
```
accuracy 계산
```python
score = tf.model.evaluate(x_data,y_data)
print("accuracy: ",score[1])
```
accuracy 결과
```python
# accuracy:  1.0
```
