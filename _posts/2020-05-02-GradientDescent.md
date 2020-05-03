---
layout: post
title: Gradient Descent
description: "Gradient Descent code"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---

```python
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(0) #random seed 초기화

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]
w_data = []
cost_data = []
w = tf.Variable(tf.random.normal([1], -100., 100.)) #-100과 100사이의 랜덤한 숫자를 shape[1]로 선택함

for step in range(1000+1):
    hypothesis = w*x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    learning_rate = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(w,x_data)-y_data, x_data))
    descent = w - tf.multiply(learning_rate, gradient)
    w.assign(descent)

    if step%100 == 0:
        print("%d | %f | %f"%(step, cost.numpy(), w.numpy()))
    w_data.append(w.numpy())
    cost_data.append(cost.numpy())

plt.scatter(w_data, cost_data)
plt.show()

```
![image](/assets/GDCgraph.png)
