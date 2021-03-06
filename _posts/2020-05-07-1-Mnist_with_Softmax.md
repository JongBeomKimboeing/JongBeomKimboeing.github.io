---
layout: post
title: Mnist with softmax
description: " Mnist with softmax"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---
# Softmax 단일 계층을 이용하여 mnist를 분류했다.
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
import time

start = time.time()

learning_rate = 0.001
batch_size = 100
training_epoch = 15
nb_classes = 10

mnist = tf.keras.datasets.mnist #mnist 데이터셋 객체 생성

(x_train, y_train), (x_test, y_test) = mnist.load_data() #mnist 데이터 load 해오기 (numpy array의 tuple형이 반환 됨)
x_train, x_test = x_train/255.0, x_test/255.0

print(x_train.shape) #(60000,28,28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) #(60000, 784)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])  #(10000, 784)
print(x_train.shape)

#y를 one hot encoding으로 변환
#3개의 class를 갖고 5개의 답이 들어있는 array([0,2,1,2,0])이 있다고 하자,
#위 array를 'to_categorical'로 변환하면
'''array([[1., 0., 0.],
          [0., 0., 1.],
          [0., 1., 0.],
          [0., 0., 1.],
          [1., 0., 0.]], dtype=float32)
로 변환된다
'''
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=10, input_dim=784, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.summary()

history = tf.model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch)


predictions = tf.model.predict(x_test)
print("prediction: \n",predictions)
score = tf.model.evaluate(x_train, y_train)
print("accuracy: ",score[1])
print("time: %d sec" %(time.time() - start))

r = random.randint(0, x_test.shape[0]-1)
print("Label:", np.argmax(y_test[r]))
print("predicted:", np.argmax(predictions[r]))
plt.imshow(x_test[r].reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()
```
