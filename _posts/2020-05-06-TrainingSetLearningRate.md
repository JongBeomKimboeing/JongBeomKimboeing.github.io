---
layout: post
title: Training set과 test set의 구분, 적절한 learning rate
description: "Training set과 test set의 구분, 적절한 learning rate"
modified: 2020-05-06
tags: [김성훈,DL]
categories: [김성훈DL]
---
```python
import tensorflow as tf

#training dataset
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
#test dataset은 training dataset과 따로 분리해서 사용한다.
#대개의 경우 trainingset의 20%를 test dataset으로 분리해서 사용한다.
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

#learning_rate = 65535 #learning_rate이 너무 커서 cost가 0으로 수렴하지 않고 무한을 발산해 버린다.
learning_rate = 0.1 #적절한 learning rate
#learning_rate = 1e-10 # learning_rate이 너무 작아서 cost가 0으로 수렴하지 않고 local minimum에 멈추거나 특정 값에 멈춰버린다.

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=['accuracy'])
tf.model.fit(x_data, y_data, epochs=1000,verbose=1)

print("prediction: ",tf.model.predict_classes(x_test))

print("Acuuracy: ",tf.model.evaluate(x_test, y_test)[1])
#tf.model.evaluate의 [0]는 loss value를 가지고 있고, [1]은 tf.model.compile에서 정의한 metrics 값을 가지고 있음

```
