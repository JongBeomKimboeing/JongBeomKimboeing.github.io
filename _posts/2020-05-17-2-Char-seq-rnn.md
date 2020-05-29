---
layout: post
title: 간단한 문장 RNN 학습시키기
description: "간단한 문장 RNN 학습시키기"
modified: 2020-05-17
tags: [김성훈,DL]
categories: [김성훈DL]
---
# 간단한 문장을 RNN을 이용하여 학습시켜본다.
```python
import tensorflow as tf
import numpy as np

sample = "if you want you"

idx2char = list(set(sample)) #sample을 set자료형으로 변환시킨 뒤 list형으로 변환
# 각각의 문자를 하나씩 중복없이 뽑아내어 index를 준다.
char2idx = {c: i for i,c in enumerate(idx2char)}
# idx2char을 하나씩 뽑아와서 문자와 인덱스를 가져오고, 이를 이용하여 딕셔저리 형태로 만들어준다.
sample_idx = [char2idx[c] for c in sample]
# sample 문자열의 문자를 하나씩 가져와서 char2idx 딕셔너리에서 불러와 인덱스 형태로 만들어준다.
x_data = [sample_idx[:-1]] #sample의 (0 ~ n-1)까지의 데이터를 x_data로 이용한다.
y_data = [sample_idx[1:]] #sample의 (1 ~ n)까지의 데이터를 y_data로 이용한다.

dic_size = len(char2idx) # RNN input size (one hot size)
hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) #final output size (RNN or softmax, etc.) -> 10
batch_size = 1 #one sample data, one batch
sequence_length = len(sample) - 1 # number of lstm rollings(number of unit) -> 15 - 1 = 14
learning_rate = 0.1

x_one_hot_eager = tf.one_hot(x_data, num_classes) # shape-> (1, 14, 10)
x_one_hot_numpy = tf.keras.utils.to_categorical(x_data, num_classes) # numpy array로 one_hot 형성
y_one_hot_eager = tf.one_hot(y_data,num_classes)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.LSTM(units=num_classes, input_shape=(sequence_length, x_one_hot_eager.shape[2]), return_sequences= True))
tf.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units= num_classes, activation='softmax')))
tf.model.summary()
tf.model.compile(loss= 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
tf.model.fit(x_one_hot_eager, y_one_hot_eager, epochs=50)

predictions = tf.model.predict(x_one_hot_eager)

for i, prediction in enumerate(predictions):
    result_str = [idx2char[c] for c in np.argmax(prediction, axis=1)]
    print("\tPrediction str: ", ''.join(result_str))
```
