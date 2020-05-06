---
layout: post
title: softmax를 이용하여 동물데이터 분류
description: "softmax를 이용하여 동물데이터 분류"
modified: 2020-05-06
tags: [김성훈,DL]
categories: [김성훈DL]
---

## 아래 코드는 김성훈 강좌를 실습했으나, 계속 애러가 나서 실패한 코드이다.
```python
import tensorflow as tf
import numpy as np

xy = np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = np.array(xy[:,[-1]], dtype=np.int64)
nb_classes = 7

y_one_hot = tf.one_hot(list(y_data), nb_classes)
# list(y_data)->ndarray형의 데이터를 list형의 데이터로 변환
# y_data에 onehot을 해주면 shape=(?,1,7)
# ex)[[0],[3]] -> shape = (2,1) => rank=2
#    onehot을 해주면 [[[1,0,0,0,0,0,0]],[[0,0,0,1,0,0,0]]] -> shap(2,1,7) =>rank=3
#    다음과 같이 onehot을 해주면 rank(차원 수)가 하나 추가됨
#    [[[0의 onehot값]]] -> 원래 [[]]에 0의 onehot값이 list로 추가됨
#그러므로 reshape을 하여 rank를 1만큼 감소시켜야 함.
y_one_hot = tf.reshape(y_one_hot,[-1,nb_classes])
#shape=(?,7)

w = tf.Variable(tf.zeros([16, nb_classes]),name='weight')
b = tf.Variable(tf.zeros([nb_classes]),name='bias')

variables = [w,b]

def logit_fn(x):
    return tf.matmul(x,w)+b

def hypothesis(x):
    tf.nn.softmax(logit_fn(x))

def cost_fn(x,y):
    #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis(x)), axis=1))
    logits = logit_fn(x)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cost = tf.reduce_mean(cost_i)
    return cost

def grad_fn(x,y):
    with tf.GradientTape() as tape:
        loss = cost_fn(x,y)
        grads = tape.gradient(loss, variables)
        return grads

def prediction(x,y):
    pred = tf.argmax(hypothesis(x),1)
    #1은 axis의 방향을 의미하며, 함수의 의미는 hypothesis matrix의 y축 방향에서 가장 큰 값의 index를 구해 return한다.
    correct_prediction = tf.argmax(tf.equal(pred, tf.argmax(y,1)))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def train(x,y,epoch=1000, print_time=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epoch):
        grads = grad_fn(x,y)
        optimizer.apply_gradients(zip(grads,variables))
        if (i==0) | (i+1%print_time==0):
            acc = prediction(x,y).numpy()
            loss = cost_fn(x,y).numpy()
            print("steps: %d Loss: %f ACC:%f"%(i+1, loss, acc))

train(x_data, y_one_hot)
```
## 위 코드를 대신하여 김성훈 github에 있는 코드를 참조하였다.
### 좀 더 tensorflow2.0 다운 코드이다.
```python
import tensorflow as tf
import numpy as np

xy = np.loadtxt("zoo.csv", delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7

y_one_hot =tf.keras.utils.to_categorical(y_data, nb_classes) #y_data를 nb_classes 개수만큼 one_hot을 만들어준다.
print("one_hot:",y_one_hot)
tf.model = tf.keras.Sequential() #layer 형성
tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation='softmax'))
#tf.model.add : layer 추가
#tf.keras.layers.Dense: layer 설정 -> units: 출력값 개수, input_dim: 입력값 개수, activation: 사용할 activation function
tf.model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
#tf.model.compile: 훈련을 위한 모델 구성-> loss: loss function설정, optimizer: optimizer 설정,
# metrics: 훈련과 테스팅동안 게산할 측정 항목 (ex) accuracy, mse(mean square error))
tf.model.summary() #형성한 network의 정보를 출력

#epoch만큼 모델을 훈련시킴
history = tf.model.fit(x_data,y_one_hot,epochs=1000)

test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])# expected prediction == 3 features를 test_data로 준다.
print(tf.model.predict(test_data), tf.model.predict_classes(test_data))
# tf.model.predict: test 샘플에 대한 예측값을 출력
# tf.model.predict_classes: 샘플에서 예측한 값 중 가장 큰 값의 위치(class)를 출력

pred = tf.model.predict_classes(x_data)
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
```
