---
layout: post
title: Practice of the applications of array
description: "Practice of the applications of array"
modified: 2020-05-07
tags: [김성훈,DL]
categories: [김성훈DL]
---
# array 응용하기
```python
import tensorflow as tf
import numpy as np
tf.debugging.set_log_device_placement(True)
```

# rank, shape, axis
## 1. rank
rank: 차원수 (array.ndim로 확인 가능)
-> '[ '가 몇 개 있는 지로 쉽게 파악가능하다.<br>
ex) [0., 1., 2., 3., 1., 5., 6.] -> rank: 1<br>
ex) [[0., 1., 2., 3., 1., 5., 6.]] -> rank: 2<br>
## 2. shape
shape 결정 방법: rank가 n이면 shape 안의 요소 개수가 n (즉, rank=3 shape=(?,?,?), rank=4 shape=(?,?,?,?))<br>
shape 요소 결정 방법: [[1,2],[3,4]] -> 가장 안쪽 '[]'가 품은 요소의 개수가 shape의 마지막에 옴 (2,2)<br>
ex) [[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],<>br>
    [[13,14,15,16],[17,18,19,20],[21,22,23,24]]]] -> rank:4, shape(1,2,3,4) ('[]'가 품은 요소의 개수를 차례로 보면 됨)<br>
## 3. axis
axis: 가장 안 쪽에 있는 '[]'가 가장 큰 axis 값<br>
(axis=-1은 가장 큰 axis 값에 해당하는 axis를 의미함.)<br>
<br>
ex) [[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],<br>
    [[13,14,15,16],[17,18,19,20],[21,22,23,24]]]]<br>
<br>
위 array의 axis를 분석하면 아래와 같다.
<br>

[1,2,3,4]-> axis=3<br>

[[1,2,3,4],[5,6,7,8],[9,10,11,12]]->axis=2<br>
<br>
[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],<br>
[[13,14,15,16],[17,18,19,20],[21,22,23,24]]] axis=1<br>
<br>
[[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],<br>
[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]] axis=0<br>
```python
#print("rank, shape, axis")
t = np.array([0., 1., 2., 3., 1., 5., 6.])
print(t)
print(t.ndim)
print(t.shape) # shape(array의 모양) (결과: (7,))
```

```python
t1 = np.array([[1.,2.,3.],[4.,5.,6.],[7., 8., 9.],[10., 11., 12.]])
print(t1)
print(t1.ndim) # rank: 2
print(t1.shape) # shape: (4,3)
print('\n')
```

```python
t1 = np.array([[1.,2.,3.],[4.,5.,6.],[7., 8., 9.],[10., 11., 12.]])
print(t1)
print(t1.ndim) # rank: 2
print(t1.shape) # shape: (4,3)
print('\n')
```
## 4. slicing
t[0], t[1], t[-1]: 자리에 해당하는 원소 (결과: 0.0 1.0 6.0) **-1은 마지막 원소를 말함<br>
<br>
t[2:5], t[4:-1]: [시작:끝] ->"t[시작] ~ t[끝-1]" 을 출력한다.([2. 3. 1.] [1. 5.])<br>
<br>
t[:2], t[3:]: [:끝] -> t[0] ~ t[끝-1] // [시작:] -> t[시작] ~ t[끝] (결과: [0. 1.] [3. 1. 5. 6.])<br>
```python
t = np.array([0., 1., 2., 3., 1., 5., 6.])
print(t[0], t[1], t[-1]) # 자리에 해당하는 원소 (결과: 0.0 1.0 6.0) **-1은 마지막 원소를 말함
print(t[2:5], t[4:-1]) # [시작:끝] ->"t[시작] ~ t[끝-1]" 을 출력한다.([2. 3. 1.] [1. 5.])
print(t[:2], t[3:]) # [:끝] -> t[0] ~ t[끝-1] // [시작:] -> t[시작] ~ t[끝] (결과: [0. 1.] [3. 1. 5. 6.])
```

# matmul vs multiply
1) matmul: matrix의 내적을 계산 -> shape을 내적 가능한 형태로 반드시 맞춰줘야 한다.<br>

2) multiply(*): matrix의 element wise 곱 계산<br>

```python
#print("matmul vs multiply")
matrix1 = tf.constant([[1.,2.],[3.,4.]])
matrix2 = tf.constant([[1.],[2.]])
print("matrix1 shape", matrix1.shape)
print("matrix2 shape", matrix2.shape)
print(tf.matmul(matrix1,matrix2))
print(matrix1*matrix2)
print('\n')
```

# broadcasting
shape, rank가 맞지 않은 두 matrix를 연산할 때 broadcasting이 사용된다.
(주석/코드 참고)
```python
print("broadcasting")
matrix3 = tf.constant([[3.,3.]])
matrix4 = tf.constant([[2.,2.]])
print((matrix3+matrix4))
matrix3 = tf.constant([[1.,2.]])
matrix4 = tf.constant(3.) #rank,shape이 모두 다른 (3.)이 [[3.,3.]]으로 broadcasting 되어 연산된다.
print((matrix3+matrix4))
matrix3 = tf.constant([[1.,2.]])
matrix4 = tf.constant([3.,4.]) # rank가 다른 [3.,4.]이 [[3.,4.]]으로 broadcasting 되어 연산된다.
print((matrix3+matrix4))
matrix3 = tf.constant([[1.,2.]]) # shape가 다른 [[1.,2.]]이 [[1.,2.],[1.,2.]]으로 broadcasting 되어 연산된다.
matrix4 = tf.constant([[3.],[4.]]) # shape가 다른 [[3.],[4.]]이 [[3.,3.],[4.,4.]]으로 broadcasting 되어 연산된다.
print((matrix3+matrix4))
print('\n')
```
# reduce_mean
전체의 요소들에 대한 평균값을 구한다.<br>
<br>
1)
```python
a = tf.reduce_mean([1,2],axis=0) # tensor(배열)의 요소가 정수여서 결과 값으로 정수를 내놓음.
                                 # axis=0은 [1,2]를 의미한다.
print(a)
```
2)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_mean(x) # x 전체의 요소들에 대한 평균값을 구한다. -> 결과: 2.5
print(a)
```
3)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_mean(x, axis=1) #axis=1은 [1.,2.]의 평균과 [3.,4.]의 평균 각각의 축에 대한 평균을 구한다. -> 결과; [1.5, 3.5]
print(a)
```
4)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_mean(x, axis=-1) #axis=-1은 가장 큰 axis (여기서는 axis=1)와 같다. -> 결과; [1.5, 3.5]
print(a)
```
5)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_mean(x, axis=0) #axis=0은 [[1.,2.],[3.,4.]]의 평균을 구한다. (([1., 2.]+[3., 4.])/2 = [2., 3.]) -> 결과: [2., 3.])
print(a)
print('\n')
```
# reduce_sum
전체의 요소들에 대한 합을 구한다.<br>
<br>
1)
```python
print("reduce_sum")
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_sum(x) # x 전체의 요소들에 대한 합을 구한다. -> 결과: 10.0
print(a)
```
2)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_sum(x, axis=1) #axis=1은 [1.,2.]의 합과 [3.,4.]의 합 각각의 축에 대한 합을 구한다. -> 결과; [3., 7.]
print(a)
```
3)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_sum(x, axis=-1) #axis=-1은 가장 큰 axis (여기서는 axis=1)와 같다. -> 결과; [3., 7.]
print(a)
```
4)
```python
x = [[1., 2.],
     [3., 4.]]
a = tf.reduce_sum(x, axis=0) #axis=0은 [[1.,2.],[3.,4.]]의 합을 구한다. (([1., 2.]+[3., 4.]) = [3., 6.]) -> 결과: [4., 6.])
print(a)
print('\n')
```
# reduce_mean과 reduce_sum
```python
print("reduce_mean과 reduce_sum")
a = tf.reduce_mean(tf.reduce_sum(x, axis=-1))
print(a)
print('\n')
```
