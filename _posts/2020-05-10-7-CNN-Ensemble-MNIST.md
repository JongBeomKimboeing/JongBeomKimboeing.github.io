---
layout: post
title: CNN with Ensemble
description: "CNN with Ensemble"
modified: 2020-05-10
tags: [김성훈,DL]
categories: [김성훈DL]
---
# 4개의 다른 CNN 모델을 Ensemble 시켜 학습
```python
import tensorflow as tf
import numpy as np
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 225
x_test = x_test / 225
print(x_train.shape) #(60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0] ,28 ,28 ,1) # (60000, 28, 28, 1) #6000개의 데이터를 28*28의 이미지로 변환하고, 색은 하나이다.
print(x_train.shape)
print(x_test.shape) #(10000, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print(x_test.shape) # (10000, 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

learning_rate = 0.001
training_epoch = 10
batch_size = 128
drop_out_rate = 0.5

#layer1
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
# tf.keras.layers.Conv2D-> filters= : fiter의 개수, kernel_size=: kernal 크기,  input_shape=: input data의 모양
#                       -> strides=(1, 1): filter를 얼마만큼의 stride로 움직일 것인가.
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer2
tf.model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size=(3,3), activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer3
tf.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding="same",activation='relu'))
tf.model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#layer3(Fully connected)
tf.model.add(tf.keras.layers.Flatten())
tf.model.add(tf.keras.layers.Dense(units=200,kernel_initializer='glorot_normal',activation='relu'))
#layer4(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=200,kernel_initializer='glorot_normal',activation='relu'))
#layer5(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=100,kernel_initializer='glorot_normal',activation='relu'))
#layer(Fully connected)
tf.model.add(tf.keras.layers.Dropout(0.5))
tf.model.add(tf.keras.layers.Dense(units=10,kernel_initializer='glorot_normal', activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
tf.model.summary()
```
## ensemble 학습을 추가한 부분
ensmeble은 각각의 모델로 부터 나온 예측값을 합하여<br>
합한 예측값으로 최종 예측을 하는 방법이다.<br>
```python
#----------------------------------------------------------------------------
# 4개의 CNN모델 형성

models = []
num_models = 4 
for m in range(num_models):
    models.append(tf.model)

predictions = np.zeros([y_test.shape[0], 10])

# 각각의 model로부터 나온 prediction을 합한다.

for m_idx, m in enumerate(models):
    m.fit(x_train,y_train,batch_size=batch_size,epochs=training_epoch)
    p = m.predict(x_test)
    predictions += p
    evaluation = m.evaluate(x_test, y_test)
    print(m_idx+1, 'loss: ', evaluation[0])
    print(m_idx+1, 'accuracy', evaluation[1])
print("\n")

# ensemble한 결과의 accuracy를 계산한다.
ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_test, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', ensemble_accuracy.numpy())
print("\n")
#--------------------------------------------------------------------------------
```

```python
for x in range(0, 10):
    random_index = random.randint(0, x_test.shape[0]-1)
    print("index: ", random_index,
          "actual y: ", np.argmax(y_test[random_index]),
          "predicted y: ", np.argmax(predictions[random_index]))
```
# 결과
## ensemble accuracy
```python
Ensemble accuracy: 0.9917
```
## 10개의 random한 test set에 대한 label과 prediction 비교
```python
index:  3263 actual y:  8 predicted y:  8
index:  1621 actual y:  0 predicted y:  0
index:  2522 actual y:  3 predicted y:  3
index:  2555 actual y:  3 predicted y:  3
index:  6687 actual y:  4 predicted y:  4
index:  9465 actual y:  5 predicted y:  5
index:  5064 actual y:  7 predicted y:  7
index:  3389 actual y:  8 predicted y:  8
index:  5275 actual y:  5 predicted y:  5
index:  5993 actual y:  0 predicted y:  0
```
# 고찰
위 모델은 ensemble을 적용하였다.<br>
training epoch가 10이고, model이 4개이므로, 총 40번의 epoch가 돌아간다.<br>
accuracy가 99.17%인 것을 보아, ensemble학습의 효과를 실감할 수 있었다.<br>
ensemble학습을 직접 구현해보면서 Deep Learning의 재미를 다시 한번 느꼈다.

### 실제로 ensemble하는 지 관찰
관찰을 하기 위해 2개의 model을 생성시켜 ensemble 결과를 출력했다.
#### model 1 의 prediction
```python
#1
[[9.59904867e-08 3.95651296e-06 1.72893651e-05 ... 9.99922633e-01
  1.13546082e-06 4.89888007e-05]

loss: 0.0541 - accuracy: 0.9690
1 loss:  0.10699547047736123
1 accuracy 0.969
```
#### model 2 의 prediction

```python
2
[3.6390876e-10 9.0866727e-09 7.1364738e-08 ... 9.9999964e-01
  2.2209652e-09 2.7971430e-07]

loss: 0.0348 - accuracy: 0.9797
2 loss:  0.0694328069977695
```
#### model 1과 model 2의 ensemble 결과
```python
[9.63543954e-08 3.96559963e-06 1.73607299e-05 ... 1.99992228e+00
  1.13768179e-06 4.92685150e-05]
  ```
  위 결과에서 model 1 과 model 2가 예측한 값의 첫번째 원소를 보면<br>
  두 원소가 실제로 더해져서 ensemble 되는 것을 확인할 수 있다.<br>
  위 결과를 통해, ensemble을 잘 만들었음을 확인할 수 있었고<br>
  ensemble 과정을 더 확실하게 관찰 할 수 있었다.<br>
  


