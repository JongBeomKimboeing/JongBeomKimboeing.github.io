---
layout: post
title: RNN을 이용하여 주식 예측하기
description: "RNN을 이용하여 주식 예측하기"
modified: 2020-05-17
tags: [김성훈,DL]
categories: [김성훈DL]
---
# 간단한 csv파일을 이용하여 주식예측 RNN을 만들어보기
RNN은 sequence한 데이터를 학습하는 데 주로 사용된다.
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

seq_length = 7
data_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

xy = np.loadtxt('stock.csv', delimiter=',')
xy = xy[::-1] # reverse the order -> 맨 아래있는 데이터가 맨 위로 온다.

train_size = int(len(xy)*0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:] # 'train_size - seq_length:'로 하는 이유는 test를 하기 위한 7개의 input으로 이용하여 output의 결과를 보기 위해서.

train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        x = time_series[i: i+seq_length, :]
        y = time_series[i+seq_length,[-1]]
        print(x, "->", y)
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set,seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=1, input_shape=(seq_length,data_dim)))
tf.model.add(tf.keras.layers.Dense(units=output_dim, activation='tanh'))
tf.model.summary()
tf.model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
tf.model.fit(trainX, trainY, epochs= iterations)

test_predict = tf.model.predict(testX)

plt.plot(testY)
plt.plot(test_predict)
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()
```
