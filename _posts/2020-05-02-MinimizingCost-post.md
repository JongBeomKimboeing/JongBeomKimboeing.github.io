---
layout: post
title: Minimizing Cost with python
description: " Minimizing Cost with python code"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---

```python
#cost function with pure python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3])
y = np.array([1,2,3])
fw = []
cc = []

def cost_function(w,x,y):
    cost = 0
    for i in range(len(x)):
        cost += (w*x[i] - y[i])**2
    return cost/len(x)

for feed_w in np.linspace(-3,5, num=15):  #feed_w 값을 -3~5를 15구간으로 나눈다
    curr_cost = cost_function(feed_w, x, y) #feed_w 값에 따라서 cost가 얼마나 나오는가
    print("%f | %f " %(feed_w, curr_cost))
    fw.append(feed_w)
    cc.append(curr_cost)

plt.plot(fw, cc)
plt.show()
```
![image](/blob/master/images/MinimizingCost.png)
