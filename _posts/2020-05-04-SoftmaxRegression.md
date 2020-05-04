---
layout: post
title: Gradient Descent
description: "Gradient Descent code"
modified: 2020-05-04
tags: [김성훈,DL]
categories: [김성훈DL]
---
# axis = ?
## axis=0
matrix의 x축 방향(행)
## axis=1
matrix의 y축 방향(열)
## axis=2
matrix의 z축 방향(차원)



```python
import tensorflow as tf
tf.random.set_seed(777)  # for reproducibility
import numpy as np

x_data = [[1, 2, 1, 1], #[입력의 특징들] ex) 머리1, 다리2, 코1,입1
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [[0, 0, 1], #[종류 분류] ex)괴물 닭 사람 -> 첫번째 입력데이터의 정답은 사람
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

#리스트를 numpy array로 변경하고 data_type도 변경시킨다.
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

#dataset을 선언합니다.
# dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
# dataset = dataset.repeat().batch(2)

nb_classes = 3 #분류할 종류의 개수

w = tf.Variable(tf.zeros([4, nb_classes]), name='weight')
b = tf.Variable(tf.zeros([nb_classes]), name='bias')
variables = [w, b]
print(w,b)

# softmax = exp(logits) / reduce_sum(exp(logits), dim)
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X,w) +b )
print(hypothesis(x_data))

def cost_fn(X,Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits),axis=1)
    #axis=0: x축 방향으로의 원소들의 합 (행)
    #axis=1: y축 방향으로의 원소들의 합 (열)
    #axis=2: z축 방향으로의 원소들의 합
    cost_mean = tf.reduce_mean(cost)
    return cost_mean

def grad_fn(X,Y):
    with tf.GradientTape() as tape:
        loss = cost_fn(X,Y)
        grads = tape.gradient(loss, variables)
        return grads
        #grads는 loss를 w에 대해 미분한 결과와 loss를 b에 대해 미분한 결과 두개가 나옴
print(grad_fn(x_data, y_data))

#cost_function에 대한 미분을 실행하면서 w를 업데이트 시키고
# 업데이트 된 w의 위치에서의 cost_function 미분을 다시 실행하고 w를 업데이트 시킨다.
def fit(X,Y,epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X,Y)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
        if((i==0) | (i+1)%verbose==0):
            print("Loss at epoch: %d: %f"%(i+1, cost_fn(X,Y).numpy()))

fit(x_data, y_data)

#test1
sample_data = [[2,1,3,2]]
sample_data = np.asarray(sample_data, dtype=np.float32)
a  = hypothesis(sample_data)
print(a)
print(tf.argmax(a,1))
#test2
b = hypothesis(x_data)
print(b)
print(tf.argmax(b, 1))
print(tf.argmax(y_data, 1))

```
