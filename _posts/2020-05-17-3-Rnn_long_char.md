---
layout: post
title: 긴 문장 RNN으로 학습시키기
description: "긴 문장 RNN으로 학습시키기"
modified: 2020-05-15
tags: [김성훈,DL]
categories: [김성훈DL]
---
# 긴문장을 RNN으로 학습해본다.
```python
import tensorflow as tf
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
print(char_set)
char_dict = {w: i for i,w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set) #len(char_set) -> 25
print(len(char_set))
sequence_length = 10 # RNN의 길이
learaning_rate = 0.1

dataX = []
dataY = []
# RNN의 길이만큼 잘라서 sentence를 입력해준다.
for i in range(0, len(sentence)-sequence_length): # 반복 횟수는 입력할 10개의 문자를 제외한 수만큼 반복해야한다.
    x_str = sentence[i:i+sequence_length] # 0-> 0~10, 1-> 1~11, 2-> 2~12 .....len(sentence)-1 까지
    y_str = sentence[i+1:i+sequence_length+1] # 1부터 시작해서 len(sentence)까지(끝까지)
    print(i,x_str,'->',y_str)

    x = [char_dict[c] for c in x_str]
    y = [char_dict[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) # len(dataX)-> 170

x_one_hot = tf.one_hot(dataX, num_classes)
y_one_hot = tf.one_hot(dataY, num_classes)

print(x_one_hot.shape)# result: (170, 10, 25) == (배치사이즈,RNN 길이 , 입력 수)
print(y_one_hot.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units= num_classes, input_shape=(sequence_length, x_one_hot.shape[2]),return_sequences=True))
tf.model.add(tf.keras.layers.LSTM(units=num_classes, return_sequences=True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=num_classes,activation='softmax')))
tf.model.summary()
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learaning_rate),metrics=['accuracy'])
tf.model.fit(x_one_hot,y_one_hot,epochs=100)

results = tf.model.predict(x_one_hot)

for j, result in enumerate(results):
    index = np.argmax(result,axis=1)
    if j is 0:
        print(''.join([char_set[t] for t in index]), end='') # 처음 10개는 그대로 출력을 한다.
    else:
        print(char_set[index[-1]], end='') # 11개부터는 마지막의 index를 하나하나 출력한다.
```
