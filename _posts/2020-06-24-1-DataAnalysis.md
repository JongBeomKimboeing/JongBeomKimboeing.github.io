---
layout: post
title: 데이터 분석(Numpy)
description: "데이터 분석(Numpy)"
modified: 2020-06-24
tags: [Numpy]
categories: [Numpy]
---
# Numpy

## 1. numpy 란 무엇인가?
Numerical Python의 약자로 python에서 대규모 다차원 배열을 다룰 수 있게 도와주는 라이브러리

<pre>
데이터의 대부분은 숫자 배열로 이루어져 있다.
-> 숫자 배열을 다루는 데, python list보다 numpy를 이용하는 이유는
   numpy는 list에 비해서 빠른 연산을 지원하고 메모리를 효율적으로 사용하기 때문에 이용한다.
</pre>

<br>
<br>

## 2. numpy 배열 만들기

```python
import numpy as np
print(np.array([1,2,3,4,5])) #결과: [1 2 3 4 5] -> array 객체가 반환 됨
print(np.array([3, 1.4, 2, 3, 4]))#[3.  1.4   2.  3.  4. ] ->숫자 하나가 소수이면 모두 소수로 변환
print(np.array([[1,2],[3,4]])) #2차원 array
'''
[[1 2]
 [3 4]]
'''
```

<br>
<br>


## 3. numpy 배열의 데이터타입
- int -> 정수형 타입 (i, int_, int32, int64, i8) #(i8은 64bit임)  
- float -> 실수형 타입 (f, float_, float32, float64, f8) #(float_, f8 둘 다 64bit)  
- str -> 문자열 타입 (str, U, U32)  
- bool -> 부울 타입 (?, bool_)  

<br>
<br>


## 4. numpy 배열 형변환
### 1) dtype을 통해 형변환
print(np.array([1,2,3,4], dtype='float')) #[1. 2. 3. 4.] -> 소수로 형변환

### 2) astype을 통해 형변환
print(arr.astype(int)) #[1 2 3 4] ->astype으로 형변환을 한다.

### 3) numpy 자료형 확인
print(arr.dtype) 

```python
print(np.array([1,2,3,4], dtype='float')) #[1. 2. 3. 4.] -> 소수로 형변환

# numpy는 python의 list와 달리 array에 들어있는 data type이 모두 같아야한다.
arr = np.array([1, 2, 3, 4], dtype=float)
print(arr) #[1. 2. 3. 4.]
print(arr.dtype) #float64 numpy 자료형 확인
print(arr.astype(int)) #[1 2 3 4] ->astype으로 형변환을 한다.
```

<br>
<br>

## 5. 다양한 numpy 배열 만들기

### 1) 0 또는 1로 이루어진 numpy 배열 만들기

```python
print(np.zeros(10, dtype=int)) #[0 0 0 0 0 0 0 0 0 0] -> int형으로 10개의 data를 가짐
print(np.ones((3, 5), dtype=float))
'''
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
'''
```

### 2) 다양한 방법으로 numpy 배열 만들기

- np.arange(start, end , step)) : start부터 end까지 step만큼 띄면서 만들어주라
- np.linspace(0,1,5) : 0부터 1까지의 값을 5개의 step으로 나누어 만들기
- np.random.random((2,2)) : random으로 shape 만큼 배열 만들어주기
- np.random.normal(평균, 표준편차, shape) : 정규분포를 이용하여 데이터 추출
- np.random.randint(0, 10, (2,2)) : 0이상 10미만의 값중 값을 뽑아 2x2 행렬만들기


```python
print(np.arange(0,20,2)) #[ 0  2  4  6  8 10 12 14 16 18] -> 0부터 20까지 2씩 증가시키면서 array 만들기
#np.arange(start, end , step)) #start부터 end까지 step만큼 띄면서 만들어주라
print(np.linspace(0,1,5)) #[0.   0.25 0.5  0.75 1.  ] -> 0부터 1까지의 값을 5개의 step으로 나누어 만들기

print(np.random.random((2,2))) #튜플로 shape를 넘겨준다.
'''
[[0.35240381 0.95725824]
 [0.24424917 0.79643797]]
'''
# 정규분포를 이용하여 데이터 추출
print(np.random.normal(0, 1, (2,2))) # 평균이 0, 표준편차가 1인 데이터를 2*2행렬로 만들기
'''
np.random.normal(평균, 표준편차, shape)

[[ 0.71269842  1.77530948]
 [ 0.01731875 -0.55388326]]
'''

print(np.random.randint(0, 10, (2,2))) #0이상 10미만의 값중 값을 뽑아 2*2 행렬만들기
'''
[[1 6]
 [9 0]]
'''
```

<br>
<br>

## 6. numpy library에서 자주 사용되는 함수

<pre>
np.array - 배열생성
np.zeros - 0이 들어있는 배열 생성
np.ones - 1이 들어있는 배열 생성
np.empty - 초기화가 없는 값으로 배열을 반환
np.arange(n) - 배열 버전의 range 함수
np.random - 다양한 난수가 들어있는 배열 생성
</pre>


<br>
<br>

## 7. numpy 배열의 속성

- x2.ndim : 차원
- x2.shape : shape
- x2.size : size (안에 원소가 몇개인가)
- x2.dtype : data type

```python
x2 = np.random.randint(10, size=(3,4))
print(x2)
'''
[[1 6 6 5]
 [7 8 5 2]
 [6 7 4 5]]
'''
#차원
print(x2.ndim) # 2
#shape
print(x2.shape) # (3, 4)
#size (안에 원소가 몇개인가)
print(x2.size) # 12
#data type
print(x2.dtype) # int32
```

<br>
<br>

## 8. 찾고 잘라내기

### 1) indexing

cf) np.arange(7) -> 7전까지 생성(즉, [0,1,2,3,4,5,6] 생성)

```python
x = np.arange(7)

print(x[3]) # 3

# print(x[7]) # IndexError: index 7 is out of bounds for axis 0 with size 7

x[0] = 10 # [10  1  2  3  4  5  6]

print(x)
```

### 2) slicing

- x[start:end] : start에서 end-1까지의 수 slicing
- x[::2] : 처음부터 끝까지 2씩 건너 띄면서 값 추출

```python
x = np.arange(7)
print(x) # [0 1 2 3 4 5 6]
print(x[1:4]) # [1 2 3]  ->  x[start:end] : start에서 end-1까지의 수 slicing
print(x[1:]) # [1 2 3 4 5 6] -> 1부터 끝가지
print(x[:4]) # [0 1 2 3] -> 처음부터 4 전까지
print(x[::2]) #[0 2 4 6] -> 처음부터 끝까지 2씩 건너 띄면서 값 추출
```

<br>
<br>

## 9. 내용 정리 문제

ex1)

```python
import numpy as np

print("1차원 array")
array = np.arange(10)
print(array)

# Q1. array의 자료형을 출력해보세요.
print(type(array))

# Q2. array의 차원을 출력해보세요.
print(array.ndim)

# Q3. array의 모양을 출력해보세요.
print(array.shape)

# Q4. array의 크기를 출력해보세요.
print(array.size)

# Q5. array의 dtype(data type)을 출력해보세요.
print(array.dtype)

# Q6. array의 인덱스 5의 요소를 출력해보세요.
print(array[5])

# Q7. array의 인덱스 3의 요소부터 인덱스 5 요소까지 출력해보세요.
print(array[3:6])
```

ex2)

```python
import numpy as np

print("2차원 array")
matrix = np.arange(1, 16).reshape(3,5)  #1부터 15까지 들어있는 (3,5)짜리 배열을 만듭니다.
print(matrix)


# Q1. matrix의 자료형을 출력해보세요.
print(type(matrix))

# Q2. matrix의 차원을 출력해보세요.
print(matrix.ndim)

# Q3. matrix의 모양을 출력해보세요.
print(matrix.shape)

# Q4. matrix의 크기를 출력해보세요.
print(matrix.size)

# Q5. matrix의 dtype(data type)을 출력해보세요.
print(matrix.dtype)

# Q6. matrix의 (2,3) 인덱스의 요소를 출력해보세요.
print(matrix[2,3])

# Q7. matrix의 행은 인덱스 0부터 인덱스 1까지, 열은 인덱스 1부터 인덱스 3까지 출력해보세요.
print(matrix[:2,1:4])
```


## 10. reshape & 이어붙이고 나누기

### 1) reshape: array의 shape을 변경한다.

- x.reshape(2,4) -> reshape(변경할 shape)

```python
x = np.arange(8)
print(x)
'''
[0 1 2 3 4 5 6 7]
'''
print(x.shape) # (8,)
x2 = x.reshape(2,4)
print(x2)
'''
[[0 1 2 3]
 [4 5 6 7]]
'''
print(x2.shape) # (2, 4)
```

### 2) np.concatenate: array를 이어 붙인다.

- np.concatenate 을 이용하여 이어 붙이기

```python
x = np.array([0,1,2])
y = np.array([3,4,5])
print(np.concatenate([x,y])) # [0 1 2 3 4 5]
```

- np.concatenate([],axis=) -> axis 축을 기준으로 이어붙일 수 있다.

**axis 구분이 매우 중요하다.**

```python
matrix = np.arange(4).reshape(2,2)
print(matrix)
'''
[[0 1]
 [2 3] (이 ']' 방향이 axis=0) ->]
'''
mat2 = np.concatenate([matrix,matrix],axis=0) # matrix와 matrix를 axis=0 방향(가장 바깥쪽 []을 기준)으로 이어 붙여라.
print(mat2)
'''
[[0 1]
 [2 3]
 [0 1]
 [2 3]]
'''
mat3 = np.concatenate([matrix,matrix], axis=1) # matrix와 matrix를 axis=1 방향(가장 안쪽 []을 기준)으로 이어 붙여라.
'''
[[0 1]
 [2 3]<- (이 ']' 방향이 axis=1) ]
'''
print(mat3)

'''
[[0 1 0 1]
 [2 3 2 3]]
'''
```

### 3) np.split: axis 축을 기준으로 나눌 수 있다.

- axis=0 으로 나눈 경우

```python
atrix = np.arange(16).reshape(4,4)
print(matrix)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''
upper, lower = np.split(matrix, [3], axis=0) #[3] -> 3 이전과 이후로 분리
print(upper)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
print(lower)
'''
[[12 13 14 15]]
'''
```

- axis=1 로 나눈 경우

```python
matrix = np.arange(16).reshape(4,4)
print(matrix)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
'''
left, right = np.split(matrix, [3], axis=1)
print(left)
'''
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
'''
print(right)
'''
[[ 3]
 [ 7]
 [11]
 [15]]
'''
```

- split 예시 문제

```python
import numpy as np

print("matrix")
matrix = np.array([[ 0, 1, 2, 3],
                   [ 4, 5, 6, 7],
                   [ 8, 9,10,11],
                   [12,13,14,15]])
print(matrix, "\n")

# Q1. matrix를 [3] 행에서 axis 0으로 나누기
'''
[[0  1   2  3]
 [4  5   6  7]
 [8  9  10 11]],

 [12 13 14 15]
'''
a, b = np.split(matrix, [3], axis=0)

print(a, "\n")
print(b, "\n")


# Q2. matrix를 [1] 열에서 axis 1로 나누기
'''
[[ 0]
 [ 4]
 [ 8]
 [12]],

[[ 1  2  3]
 [ 5  6  7]
 [ 9 10 11]
 [13 14 15]]
'''

c, d = np.split(matrix, [1], axis=1) # [1] 이전과 이후로 분리

print(c, "\n")
print(d)
```










































































































