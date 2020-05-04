---
layout: post
title: Gradient Descent
description: "Gradient Descent code"
modified: 2020-05-02
tags: [김성훈,DL]
categories: [김성훈DL]
---

# batch
설명
<br>
# epoch
설명
<br>
# iteration
설명
<br>
# tf.cast(조건)
설명
<br>


```python
import numpy as np
import  matplotlib.pyplot as plt
import tensorflow as tf

x_train = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
print(len(x_train))

y_train = [[0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.]]

x_test = [[5.,2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train] #x_train의 1열 성분을 추출해낸다.
x2 = [x[1] for x in x_train] #x_train의 2열 성분을 추출해낸다.
'''
colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1,x2, c=colors , marker='^')
plt.scatter(x_test[0][0],x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#그래프 그리기
'''
#x_data는 feature, y_data는 label로 이용된다.
#batchsize는 한번에 학습시킬 size로 정한다.
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
'''
tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
:기존에는 placeholder변수에 feed-dict를 통해 학습 시 데이터를 전달했다
tesorflow2.0 버전부터는 이를 이용하지 않고 입력 파이프라인을 만들어 데이터를 효율적으로 공급한다.

**dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)): x_train과 y_train은 tenosr이다.
  위 함수는 dataset을 x_train과 y_train tensor들로 초기화 해준며, tensorflow의 tensor형식( tf.Tensor() )으로 변경해준다.
  
**.batch(len(x_train)): 데이터를 batch_size만큼 나누어 준다
  ex)1000개의 데이터를 batch_size=10으로 넘겨주면 총 100개의 step(iteration)을 통해 1 epoch (모든 데이터 한 바퀴를 돈다.)를 돈다
  위에서 batch(len(x_train))은 batch(6)과 같기 때문에 그냥 1 step(iteration)을 통해 1 epoch (모든 데이터 한 바퀴를 돈다.)를 돈다.
  만약 batch(3)으로 놓을 경우 2 step(iteration)을 통해 1 epoch를 돈다.
  
  cf)batch_size를 이용하는 이유 -> 한번에 많은 양을 학습하면 학습과정이 적어지고 정확도가 높아지나,
    학습데이터의 메모리가 클 경우 모든 메모리를 한 번 학습하는게 시간이 오래 걸림, 
    그러나 batch_size를 이용하면 학습과정은 좀 많이지고 정확도가 떨어지나,
    메모리를 분할해서 학습하기 때문에 모든 메모리를 한 번 학습 시키는 데 시간이 적게 걸림.
'''

#w와 b는 학습을 통해 생성되는 모델에 쓰이는 weight과 bias
w = tf.Variable(tf.zeros([2,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

#sigmoid 함수를 나타낸 함수
#tf.sigmoid(tf.matmul(x,w)+b)와도 같음
def logistic_regression(features):
    hypothesis = tf.divide(1., 1+tf.exp(tf.matmul(features,w)+b))
    return hypothesis

#sigmoid함수의 cost function
def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(logistic_regression(features)) + (1-labels)* tf.math.log(1-hypothesis))
    return cost

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
#학습에 이용할 optimizer를 생성한다. (여기서는 GradientDescentOptimizer를 optimizer로 생성했다.)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5,dtype=tf.float32 )
    #tf.cast(조건)-> 조건에 맞으면 1.출력, 조건에 맞지 않으면 0.출력
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels),dtype=tf.int32))
    #예측과 정답의 비교를 통한 정확도 측정
    return accuracy

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels) #미분할 식
        return tape.gradient(loss_value, [w,b])
        #loss_value를 logistic_regression의 w와 b에 대해서 미분한 결과 값을 리턴

EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels in iter(dataset):
        #iter(): dataset을 batch_size만큼 나누어 iteration만큼 돌린다
        #만약 dataset의 크기가 6  batch가 3이면 iteration은 2이다.
        #dataset의 크기가 6  batch가 6이면 iteration은 1이다.
        grads = grad(logistic_regression(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [w,b]))
        #apply_gradients(): loss function을 w와 b에대해 미분한 grads를 이용하여 w와 b를 업데이트 시킴
        # -> w = w-(learning_rate)(dL/dw)loss
        #매개변수: grads_and_vars=zip(grads, [w,b]) -> 매개변수로 gradient와 업데이트시킬 변수의 튜플 리스트로 인수를 받는다.
        #zip(grads, [w,b]): zip을 쓰는 이유-> grads와 [w,b]를 튜플로 넘겨줘야해서.
        if step % 100 ==0:
            print("iter: %d, Loss: %f" %(step, loss_fn(logistic_regression(features), features, labels)))
test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print("Testset Accuracy: %f"%test_acc)
```
