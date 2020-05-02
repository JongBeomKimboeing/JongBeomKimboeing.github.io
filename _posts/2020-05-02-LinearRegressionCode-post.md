---
layout: post
title: Linear Regression
description: "Linear Regression code"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---
Linear Regression의 코드이다.
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

#data
x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

#w,b initialize
w = tf.Variable(2.9)
b = tf.Variable(0.5)
#tf.reduce_mean(배열):차원을 줄여 평균을 계산 ex)v=[1.,2.,3.,4.] -> 결과:2.5
#tf.square(값):제곱

learning_rate = 0.01 #learning rate 초기화

for i in range(1001):
    #gradient descent부분
    with tf.GradientTape() as tape:
        #. tf.GradientTape는 컨텍스트(context) 안에서 실행된 모든 연산을 테이프(tape)에 "기록"합니다.
        # 그 다음 텐서플로는 후진 방식 자동 미분(reverse mode differentiation)을 사용해
        # 테이프에 "기록된" 연산의 그래디언트를 계산합니다.
        hypothesis = w * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        # tf.reduce_mean(배열):차원을 줄여 평균을 계산 ex)v=[1.,2.,3.,4.] -> 결과:2.5
        # tf.square(값):제곱
    w_grad, b_grad = tape.gradient(cost, [w,b])
    w.assign_sub(learning_rate*w_grad) #assign_sub는 -=와 같음
    b.assign_sub(learning_rate*b_grad)
    if i%100 ==0:
        print("%d | %f | %f | %f |" %(i, w.numpy(), b.numpy(), cost))
        #텐서는 .numpy() 메서드(method)를 호출하여 넘파이 배열로 변환할 수 있다.
        #.numpy() 메서드는 텐서를 넘파이 배열로 변환합니다.
```
