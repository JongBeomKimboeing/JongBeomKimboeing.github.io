---
layout: post
title: Practice of the applications of array
description: "Practice of the applications of array"
modified: 2020-05-07
tags: [김성훈,DL]
categories: [김성훈DL]
---
# array 응용2
```python
import tensorflow as tf
import numpy as np
```
# Argmax
Argmax는 가장 큰 값의 위치를 반환한다.
1)
아래 예시에서<br>
axis=0 이므로 [0,1,2]과 [2,1,0]를 서로 비교하여 큰 요소의 위치를 출력한다.<br>
(ex)0과 2비교, 1과 1비교, 2와 0비교)-> 결과: [1,0,0]<br>
```python
print("Argmax")
x =  [[0,1,2],
      [2,1,0]]
a = tf.argmax(x, axis=0) # axis=0 이므로 [0,1,2]과 [2,1,0]를 서로 비교하여 큰 요소의 위치를 출력한다.
                         # (ex)0과 2비교, 1과 1비교, 2와 0비교)-> 결과: [1,0,0]
print(a)
```
2)
아래 예시에서<br>
axis=1 이므로 [0,1,2]에서 가장 큰요소의 위치를 출력하고, [2,1,0]에서 가장 큰 요소의 위치를 출력한다.<br>
-> 결과: [2, 0]<br>
```python
x =  [[0,1,2],
      [2,1,0]]
a = tf.argmax(x, axis=1) # axis=1 이므로 [0,1,2]에서 가장 큰요소의 위치를 출력하고, [2,1,0]에서 가장 큰 요소의 위치를 출력한다.
                         # -> 결과: [2, 0]
print(a)
```
3)
아래 예시에서<br>
axis=-1 이므로 axis=1인 경우와 같다.<br>
-> 결과: [2, 0]
print(a)
```python
x =  [[0,1,2],
      [2,1,0]]
a = tf.argmax(x, axis=-1) # axis=-1 이므로 axis=1인 경우와 같다.
                          # -> 결과: [2, 0]
print(a)
print('\n')
```
# Reshape
Reshape는 array의 shape을 변환한다.
1)
```python
print("Reshape")
t = np.array([[[0,1,2],
               [3,4,5]],
               [[6,7,8],
                [9,10,11]]])
print(t.shape) #shape(2,2,3)
a = tf.reshape(t, shape=[-1,3]) #rank를 2차원으로 줄이고, axis=1의  shape를 3으로, axis=0의 shape을 아무거나로 바꾼다.
                                # (shape에서 -1은 아무거나를 의미한다. 즉, axis=1의 값에 맞게 axis=0이 알아서 조정된다.)
```
rank를 2차원으로 줄이고, axis=1의  shape를 3으로, axis=0의 shape을 아무거나로 바꾼다.<br>
(shape에서 -1은 아무거나를 의미한다. 즉, axis=1의 값에 맞게 axis=0이 알아서 조정된다.)<br>
```python
[[ 0  1  2]
[ 3  4  5]
[ 6  7  8]
[ 9 10 11]] ->결과값
print(a)
```
2)

```python
t = np.array([[[0,1,2],
               [3,4,5]],
               [[6,7,8],
                [9,10,11]]])
a = tf.reshape(t, shape=[-1,1,3]) #rank를 3차원으로 하고, axis=2의  shape를 3으로, axis=1의 shape을 1로, axis=0을 아무거나로 바꾼다.
                                  # (shape에서 -1은 아무거나를 의미한다. 즉, axis=2와 axis=1의 값에 맞게 axis=0이 알아서 조정된다.)
print(a)
print('\n')
```
rank를 3차원으로 하고, axis=2의  shape를 3으로, axis=1의 shape을 1로, axis=0을 아무거나로 바꾼다.
(shape에서 -1은 아무거나를 의미한다. 즉, axis=2와 axis=1의 값에 맞게 axis=0이 알아서 조정된다.)
```
[[[ 0  1  2]]

 [[ 3  4  5]]

 [[ 6  7  8]]

 [[ 9 10 11]]] -> 결과값
 ```
 
# squeeze
squeeze는 한 차원을 감소시키고, 가장 안 쪽 axis를 없앤다.
```python
print("squeeze")
a = tf.squeeze([[0],[1],[2]]) #squeeze가 가장 안 쪽 axis (여기서는 axis=1)를 없애 차원을 낮춘다.
print(a)
print('\n')
#결과값: [0, 1, 2]
```

# expand_dims
expand_dims는 가장 안 쪽 axis (여기서는 axis=1) 를 추가하여 두 번째 매개변수(여기서는 1)만큼 차원을 높힌다.<br>
```python
print("expand_dims")
a = tf.expand_dims([0, 1, 2], 1) #expand_dims는 가장 안 쪽 axis (여기서는 axis=1) 를 추가하여 expand_dims의
                                 #두 번째 매개변수(여기서는 1)만큼 차원을 높힌다.
print(a)
print('\n')
#결과값: [[0],[1],[2]]
```

# one_hot
위치에 대한 원소를 주면 그 위치에 해당하는 값을 1로 하고 나머지를 0으로 한다.
```python
print("one_hot")
a = tf.one_hot([[0],[1],[2],[0]],depth=3) #depth는 class의 개수
print(a)
```
결과값<br>
```python
[[[1. 0. 0.]]

 [[0. 1. 0.]]

 [[0. 0. 1.]]

 [[1. 0. 0.]]] -> 결과값 (결과를 내놓으면서 rank가 하나 추가된다.)
 ```
 one_hot을 하게되면, rank 하나가 추가돼서 결과가 나오기 때문에,<br>
 rank를 없애는 것을 지향한다.
 ```python
a = tf.reshape(a, shape=[-1,3])
print(a)
print('\n')
```
```python
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]] -> 결과값 (추가된 rank를 없앤다.)
```
# casting
주로 tensor의 자료형 변환에 사용된다.
```python
print("casting")
a = tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32) #자료형이 float으로 돼 있는 tensor를 int로 변환한다.
print(a) #결과값 -> [1, 2, 3, 4]
a = tf.cast([True, False, 1==1,0==1], tf.int32) #자료형이 bool로 돼 있는 tensor를 int로 변환한다.
print(a)
print('\n')
#결과값 -> [1, 0, 1, 0]
```

# stack
주어진 tensor들을 쌓는다.<br>
axis=0으로 할 경우 x,y,z를 []의 형태를 차례로 쌓는다.
```python
print("stack")
x = [1, 4]
y = [2, 5]
z = [3, 6]
a = tf.stack([x,y,z], axis=0) # axis=0으로 할 경우 x,y,z를 []의 형태를 차례로 쌓는다.
print(a)
```
결과값
```python
[[1 4]   ->x
 [2 5]   ->y
 [3 6]]  ->z     (결과값)
```
axis=1로 할 경우 x,y,z의 원소별로 차례로 쌓는다.
```python
x = [1, 4]
y = [2, 5]
z = [3, 6]
a = tf.stack([x,y,z], axis=1) # axis=1로 할 경우 x,y,z의 원소별로 차례로 쌓는다.
print(a)
print('\n')
```
결과값
```python
[[1 2 3]    ->[x원소, y원소, z원소]
 [4 5 6]]   ->[x원소, y원소, z원소]
```
# ones and zeros like
주어진 tensor의 shape대로 모든 원소를 1이나 0을 채운다.<br>
1) ones_like<br>
tensor의 shape대로 모든 원소를 1로 채운다.
```python
print("ones and zeros like")
x = [[0, 1, 2],
     [2, 1, 0]]
a = tf.ones_like(x) # tensor의 shape대로 모든 원소를 1로 채운다.
print(a)
```
결과값
```python
[[1 1 1]
 [1 1 1]] ->결과값
```
2) zeros_like
tensor의 shape대로 모든 원소를 0으로 채운다.
```python
x = [[0, 1, 2],
     [2, 1, 0]]
a = tf.zeros_like(x) # tensor의 shape대로 모든 원소를 0으로 채운다.
print(a)
print('\n')
```
결과값
```python
[[0 0 0]
 [0 0 0]] ->결과값
```

# zip
복수개의 tensor를 for를 통해 한 번에 착출해낼 때 사용<br>
1)
```python
print("zip")
for x, y in zip([1,2,3],[4,5,6]):
    print(x,y)
```
결과값
```python
1 4
2 5
3 6 -> 결과값
```
2)
```python
for x, y, z in zip([1,2,3],[4,5,6],[7,8,9]):
    print(x,y,z)
```
결과값
```python
1 4 7
2 5 8
3 6 9 -> 결과값
```
