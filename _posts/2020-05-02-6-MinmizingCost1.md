---
layout: post
title: Minimizing Cost with tensorflow
description: " Minimizing Cost with tensorflow code"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---

```python
#cost funsction in Tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

def costfunction(w,x,y):
    hypothesis = w*x
    return tf.reduce_mean(tf.square(hypothesis - y))


w_value = np.linspace(-3, 5, num=15)
cost_values = []

for feed_w in w_value:
    curr_cost = costfunction(feed_w, x, y)
    cost_values.append(curr_cost)
    print("%f | %f " %(feed_w, curr_cost))

plt.plot(w_value, cost_values)
plt.show()
```
