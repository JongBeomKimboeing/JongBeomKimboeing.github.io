---
layout: post
title: Linear Regression With Normalization
description: "Linear Regression With Normalization"
modified: 2020-05-07
tags: [김성훈,DL]
categories: [김성훈DL]
---
```python
#데이터를 normalization(정규화) 한 경우 (데이터를 0과 1사이의 값들로 변환시킨다.)
#loss가 0으로 수렴한다.
import tensorflow as tf
import numpy as np

def min_max_scaler(data):
    numerator = data - np.min(data, 0) #각 행의 data들을 각 행에서 가장 작은 값으로 빼준다.
    denominator = np.max(data,0) - np.min(data, 0) #각 행의 가장 큰 값에서 작은 값을 빼준다.
    return numerator / (denominator + 1e-7) #1e-7은 0 division을 막아주기 위해서 있다.

xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

xy = min_max_scaler(xy)
print(xy)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=4))
tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.compile(loss='mse',optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()

history = tf.model.fit(x_data,y_data,epochs=1000)

prediction = tf.model.predict(x_data)
score = tf.model.evaluate(x_data,y_data)

print('Prediction: \n', prediction)
print('Cost: ', score)
```
