---
layout: post
title: 데이터 분석(Matplot)
description: "데이터 분석(Matplot)"
modified: 2020-06-24
tags: [Matplot]
categories: [Matplot]
---

# matplotlib
-> 파이썬에서 데이터를 그래프나 차트로 시각화할 수 있는 라이브러리

## 1. 간단한 그래프 그려보기

ex1)
```python
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
# 실제로 들어갈 때는 (1,1),(2,2),(3,3),(4,4),(5,5) 형태로 들어간다.
plt.plot(x,y)
plt.show()
```
ex2) <br>
- title, x축에 label, y축에 label 지정

```python
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
# 실제로 들어갈 때는 (1,1),(2,2),(3,3),(4,4),(5,5) 형태로 들어간다.
plt.plot(x,y)
plt.title("First Plot") # 그래프 상단에 title을 지정한다.
plt.xlabel("x") # x축에 label을 지정한다.
plt.ylabel("y") # y축에 label을 지정한다.
plt.show() # 그래프 보여주기
```

## 2. 객체기반 스타일로 그래프 그리기
- 객체기반 스타일 (objective oriented interface)(-> 객체기반이 더 명시적으로 설정해준다.)
-> 전 코드(state machine interface)는 자동으로 figure와 ax를 생성을 해주는 반면,<br>
객체기반에서는 figure와 ax를 손수 생성을 하여 그래프를 그린다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
fig, ax = plt.subplots()
ax.plot(x, y)#그래프 그려주기
ax.set_title("First Plot") # title 지정하기
ax.set_xlabel("x") # x label 설정
ax.set_ylabel("y") # y label 설정
plt.show() # 그래프 보여주기
```

## 3. matplotlib의 구조

<pre>
Figure: Figure는 도화지이다. (그래프를 그릴 수 있는 큰 틀) (Figure안에 다양한 그래프를 한 번에 넣을 수 있다.)
Axes: 그래프를 말한다.
Title: 그래프 이름
x_label: x축 이름
y_label: y축 이름
Grid: 그래프의 격자를 말한다.
Major tick: 큰 눈금
Minor tick: 작은 눈금
Legend: 범례 (도표의 내용을 알기 위해 본보기로 표시해 둔 기호와 부호의 설명)
</pre>

## 4. 기본적인 그래프 그리고 저장하기

- fig, ax = plt.subplots() : fig(하나의 도화지), ax(하나의 그래프를 그릴 수 있는 곳)\
- ax.plot(x,y) : 그래프 그리기
- ax.set_title("First Plot") : 제목
- ax.set_xlabel("x") : x축 이름
- ax.set_ylabel("y") : y축 이름
- fig.set_dpi(100) : 1 inch^2 당 들어가는 dot의 수 (크게 지정하면 그래프가 크게 저장이 됨)
- fig.savefig("first_plot.png") : 저장은 전체 도화지를 저장해야하므로, fig로 저장을 한다.
- plt.show() : 그래프 보여주기

```python
import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,4,5]
y = [1,2,3,4,5]
fig, ax = plt.subplots() # fig(하나의 도화지), ax(하나의 그래프를 그릴 수 있는 곳)
ax.plot(x,y) # 그래프 그리기
ax.set_title("First Plot") #제목
ax.set_xlabel("x") # x축 이름
ax.set_ylabel("y") # y축 이름
fig.set_dpi(100) # 1 inch^2 당 들어가는 dot의 수 (크게 지정하면 그래프가 크게 저장이 됨)
fig.savefig("first_plot.png") # 저장은 전체 도화지를 저장해야하므로, fig로 저장을 한다.
plt.show()
```

## 5. 여러개 그래프 그리기
-> axes를 배열형태로 나누어 그린다.
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, np.pi*4, 100) # 0 ~ 4pi 까지 100개의 구간으로 나누어 x에 저장
fig, axes = plt.subplots(2,1) # plt.subplots(2,1) 새로 축으로 0, 1 두개의 데이터를 가진 그래프를 그린다.(그래프의 개수 지정)
axes[0].plot(x, np.sin(x)) # 첫 번째 그래프를 그린다 (x 축에는 x, y 축에는 sin(x)를 넣는다.)
axes[1].plot(x, np.cos(x)) # 두 번째 그래프를 그린다 (x 축에는 x, y 축에는 cos(x)를 넣는다.)
plt.show()
```


## 6. matplotlib의 옵션들

## 1) line plot

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(15)
y = x**2
ax.plot(x, y,
        linestyle= ":", # 점선으로 표시
        marker= "*", # 중간 중간 '*'로 표시
        color= "#524FA1") # 색
plt.show()
```

### 1-1) line style

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, linestyle="-") # 일반 직선으로 그리기 ( ax.plot(x, x, linestyle="solid") 로도 표현 가능) 
ax.plot(x, x+2, linestyle="--") # 끊어진 직선으로 그리기 ( ax.plot(x, x+2, linestyle="dashed") 로도 표현 가능)
ax.plot(x, x+4, linestyle="-.") # 직선과 점으로 그리기 ( ax.plot(x, x+4, linestyle="dashdot") 로도 표현 가능)
ax.plot(x, x+6, linestyle=":") # 점선으로 그리기 ( ax.plot(x, x+6, linestyle="dotted") 로도 표현 가능)
plt.show()
```


### 1-2) color

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, color="r") # r,g,b,c,m,y,k 7개의 색상을 설정 가능하다.
ax.plot(x, x+2, color="green") # 이름을 직접 넣어도됨
ax.plot(x, x+4, color="0.8") # 0~1 사이의 값을 넣어 명암정도를 넣어준다.(0이면 검정 1이면 흰색으로 0.8은 밝은 회색이 나온다)
ax.plot(x, x+6, color="#524FA1") # r,g,b에 대한 16진수 코드도 들어갈 수 있다.
plt.show()
```


### 1-3) marker

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
fig, ax = plt.subplots()
ax.plot(x, x, marker=".") # '.'으로 표시하면 '.'이 marker가 표시 됨
ax.plot(x, x+2, marker="o") # 'o'으로 표시하면 '원'으로 marker가 표시 됨
ax.plot(x, x+4, marker="v") # 'v'로 표시하면 '역삼각형'으로 marker가 표시 됨 ('^'를 이용하면 maker가 삼각형이 됨)
ax.plot(x, x+6, marker="s") # 's'로 표시하면 '사각형'으로 marker가 표시 됨
ax.plot(x, x+8, marker="*") # '*'로 표시하면 '별'로 marker가 표시 됨
plt.show()
```

## 2) 그래프 자체에 대한 옵션

### 2-1) 축 경계 조정하기
- set_xlim을 이용하여 x축의 시작과 끝을 지정 가능하다.
- set_ylim을 이용하여 y축의 시작과 끝을 지정 가능하다.
- set_xlim과 set_ylim을 해주지 않으면 matplotlib에서 가장 최적화된 형태로 보여줌

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000) # 0 ~ 10 사이의 1000개의 데이터를 뽑아서 x로 가져온다.
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))


ax.set_xlim(-2, 12) # set_xlim을 이용하여 x축의 시작과 끝을 지정 가능하다. (여기서는 -2부터 12까지)
ax.set_ylim(-1.5, 1.5) # set_ylim을 이용하여 y축의 시작과 끝을 지정 가능하다. (여기서는 -1.5부터 1.5까지)


#set_xlim과 set_ylim을 해주지 않으면 matplotlib에서 가장 최적화된 형태로 보여줌
```

### 2-2) 범례
-> 범례란, 어느 기호에 대한 정보가 들어있는 박스이다.

<br>

- ax.legend(loc=, shadow=, fancybox=, borderpad=) 을 이용하여 범례를 만든다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, x, label='y=x') # label을 이용하여 범례를 label해준다.
ax.plot(x, x**2, label='y=x^2') # label을 이용하여 범례를 label해준다.
ax.set_xlabel("x") # x축 이름 지정
ax.set_ylabel("y") # y축 이름 지정

ax.legend(loc='upper left', # 범례를 가지고 있는 박스의 위치 (lower, upper, left, right,center 가 있다.)
          shadow=True, # 범례를 가지고 있는 박스의 그림자를 표시하는지
          fancybox=True, # 범례를 가지고 있는 박스의 모서리를 둥글게 만들기
          borderpad=2 # 범례 데이터를 가지고 있는 박스의 크기
          )
plt.show()
```

## 3) Scatter 그래프

- plot에서 "o"를 marker로 지정해주지 않으면 scatter 그래프 형식을 갖는다.(marker="o"로 할 경우 선이 나타남)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(10)
ax.plot(
    x, x**2, "o", # "o"를 marker로 지정해주지 않으면 scatter 그래프 형식을 갖는다.(marker="o"로 할 경우 선이 나타남)
    markersize=15, # scatter로 표시한 marker의 크기
    markerfacecolor='white', # marker 안쪽의 색
    markeredgecolor='blue' # marker 태두리 색
)
plt.show()
```

- 그냥 scatter를 호출하여 그릴 수도 있다.
```python
scatter = plt.scatter(x,y)
```


## 7. Bar & Histogram


















































































