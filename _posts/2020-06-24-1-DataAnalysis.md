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

## 3. numpy 배열의 데이터타입
- int -> 정수형 타입 (i, int_, int32, int64, i8) #(i8은 64bit임)  
- float -> 실수형 타입 (f, float_, float32, float64, f8) #(float_, f8 둘 다 64bit)  
- str -> 문자열 타입 (str, U, U32)  
- bool -> 부울 타입 (?, bool_)  


## 4. numpy 배열 형변환

```python
print(np.array([1,2,3,4], dtype='float')) #[1. 2. 3. 4.] -> 소수로 형변환

# numpy는 python의 list와 달리 array에 들어있는 data type이 모두 같아야한다.
arr = np.array([1, 2, 3, 4], dtype=float)
print(arr) #[1. 2. 3. 4.]
print(arr.dtype) #float64
print(arr.astype(int)) #[1 2 3 4] ->astype으로 형변환을 한다.
```




























































































































