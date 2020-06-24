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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

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

<br>
<br>

## 7. Bar & Histogram

### 1) bar plot

- ax.bar(x, x*2) : bar로 지정하여, bar 그래프를 그릴 수 있다.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10) # x 값은 0~9 사이의 값
fig, ax = plt.subplots(figsize=(12,4)) # figure size 조정(여기서는 가로 12 세로 4)
ax.bar(x, x*2) # bar로 지정하여, bar 그래프를 그릴 수 있다.
plt.show()
```

### 2) 누적 bar plot
-> bottom을 계속해서 지정해주어 쌓아 올린다.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(3)
y = np.random.rand(3) # x, y, z 모두 0 ~ 1 사이의 데이터 3개를 추출한다.
z = np.random.rand(3)

data = [x, y, z]

fig, ax = plt.subplots()
x_ax = np.arange(3) # 0,1,2를 가져온다.

#-----------------------------------------------------------------------------------------------------------------------------------
for i in x_ax: #x_ax를 하나씩 가져온다.
    ax.bar(x_ax, data[i], bottom= np.sum(data[:i], axis=0))
    # x_ax 번째 막대그래프에 data[i]번째(x,y,z)를 그리고,  bottom= np.sum(data[:i], axis=0)을 통해 data를 쌓을 시작위치를 설정해준다.
#-----------------------------------------------------------------------------------------------------------------------------------

ax.set_xticks(x_ax) # tick을 x_ax로 설정해준다.
ax.set_xticklabels(["A", "B", "C"]) # x_ax의 0,1,2를 A,B,C로 표현한다.
plt.show()
```

### 3) Histogram(도수분포표)

- ax.hist(data, bins=) 을 이용하여 그린다.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
data = np.random.randn(1000)
ax.hist(data, bins=50) # histogram을 그리기 위해 hist를 사용한다.
                       # bins: 막대기의 개수를 지정해준다.
plt.show()
```

### 4) 다양하게 그래프 그리기

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 아래는 한글 설정을 위한 코드이다.
fname= 'NanumBarunGothic.ttf'
font = fm.FontProperties(fname= fname).get_name()
plt.rcParams["font.family"] = font


x = np.array(["축구", "야구", "농구", "배드민턴", "탁구"])
y = np.array([18, 7, 12, 10, 8])
z = np.random.randn(1000)

fig, axes = plt.subplots(1, 2, figsize=(8, 4)) #그래프는 가로로 2개를 나열했고, 각각의 그래프 크기는 가로:8, 세로:4이다.

axes[0].bar(x, y) # 첫번째 그래프는 bar 형식으로 그린다.
axes[1].hist(z, bins=50) # 두번째 그래프는 histogram 형식으로 그린다.

plt.show()
```

<br>
<br>

## 8. matplot with pandas

pandas의 data frame이나 series data를 넣어서 그래프를 그려보자.

<br>
<br>

### ex1)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pokemon.csv")

fire = df[(df['Type 1'] == 'Fire') | (df['Type 2'] == 'Fire')]
water = df[(df['Type 1'] == 'Water') | (df['Type 2'] == 'Water')]

fig, ax = plt.subplots()
ax.scatter(fire['Attack'], fire['Defense'], color="R", label='Fire', marker='*', s=50)
# 점의 x위치, 점의 y위치, 색, 라벨, marker, marker의 사이즈를 정해준다.
ax.scatter(water['Attack'], water['Defense'], color="B", label='Water', s=50)
# marker가 초기화 돼 있지 않으면 원으로 표시된다.
ax.set_xlabel("Attack")
ax.set_ylabel("Defense")
ax.legend(loc="upper right")

plt.show()
```

<br>

 ### ex2)
 토끼와 거북이 경주 결과 시각화 <br>
-> 내 코드는 x축을 직접 지정해 줬다. 그러나, dataframe의 index가 x 데이터로 바로 변환된다는 점을 이용해도 된다.

```python
from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = 'NanumBarunGothic'


def main():
    # 아래 경로에서 csv파일을 읽어서 시각화 해보세요
    # 경로: "./data/the_hare_and_the_tortoise.csv"
    read = pd.read_csv("the_hare_and_the_tortoise.csv")
    x = read["시간"]
    rabbit = read["토끼"]
    turtle = read["거북이"]
    fig, ax = plt.subplots()
    ax.plot(x, rabbit, color='blue', label="rabbit")
    ax.plot(x, turtle, color='orange', label="turtle")
    ax.legend(loc="upper left")
    plt.show()
    pass

if __name__ == "__main__":
    main()
```

### ex2-1)

아래 코드는 dataframe의 column을 x로 바로 지정해주어 명시적으로 x 값을 지정해주지 않아도 된다.

```python
from matplotlib import pyplot as plt
import pandas as pd


def main():
    # 아래 경로에서 csv파일을 읽어서 시각화 해보세요
    # 경로: "./data/the_hare_and_the_tortoise.csv"
    read = pd.read_csv("the_hare_and_the_tortoise.csv", index_col=0) # 0 번째 column을 index로 사용한다.
    # read.set_index("시간", inplace=True) # 0 번째 column을 index로 사용한다.
    rabbit = read["토끼"]
    turtle = read["거북이"]
    fig, ax = plt.subplots()
    ax.plot(rabbit, color='blue', label="rabbit")
    ax.plot(turtle, color='orange', label="turtle")
    ax.legend(loc="upper left")
    plt.show()
    pass

if __name__ == "__main__":
    main()

```

### ex3) 월드컵 우승국가 시각화

(어떻게 그래프를 그리는지 참고할 )

```python
from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams["font.family"] = 'NanumBarunGothic'


# 아래 경로에서 csv파일을 읽어서 시각화 해보세요
# 경로: "./data/WorldCups.csv"
df = pd.read_csv("WorldCups.csv")    # 월드컵 정보를 담는 csv 파일을 읽어옵니다.
                                            # 어떤 자료를 갖는지 직접 확인해보세요

winners = df["Winner"] # 읽어온 데이터 프레임 중 "우승국"을 의미하는 칼럼을 가져오세요.
print(winners)
winner_dict = {}
for i in winners:
    winner_dict[i] = len(winners[i == winners])
print(winner_dict)
x = list(winner_dict.keys())
y = list(winner_dict.values())
fig, ax = plt.subplots(figsize=(8, 8))

ax.bar(x,y)
ax.set_xlabel("Country")
ax.set_ylabel("Number")
ax.set_xticks(x)

#for key, value in zip(list(winner_dict.keys()), list(winner_dict.values())):
#    ax.bar(key, value)   # -> 내가 짜본 코드


plt.show()

```

### ex3-1) 월드컵 우승국가 시각화

dictionary를 다루는 코드 부분을 변형시켰다

```python
from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams["font.family"] = 'NanumBarunGothic'


# 아래 경로에서 csv파일을 읽어서 시각화 해보세요
# 경로: "./data/WorldCups.csv"
df = pd.read_csv("WorldCups.csv")    # 월드컵 정보를 담는 csv 파일을 읽어옵니다.
                                            # 어떤 자료를 갖는지 직접 확인해보세요!
# print(df)

winners = df["Winner"]          # 읽어온 데이터 프레임 중 "우승국"을 의미하는 칼럼을 가져오세요.

# 국가 별 우승 횟수를 나타내는 딕셔너리 입니다.
winner_dict = {}


for i in winners :          # 우승국을 반복문으로 읽으며, 해당 국가의 우승 횟수를 1씩 증가시킵니다.
    if i in winner_dict :
        winner_dict[i] = winner_dict[i] + 1
        # i(우승국)이 이미 winner_dict에 있다면, value를 1 증가시킵니다.
    else :
        winner_dict[i] = 1
        # i(우승국)이 winner_dict에 최초로 등장한다면, value를 1로 설정합니다.

print(winner_dict)

X = list(winner_dict.keys())      # X축 변수, 즉 우승국을 나타냅니다.
Y = list(winner_dict.values())    # Y축 변수, 즉 우승 횟수를 나타냅니다.

fig, ax = plt.subplots(figsize=(8, 8))

# ax.plot(X, Y)

ax.bar(X, Y)

ax.set_xlabel("Country")
ax.set_ylabel("Number")

ax.set_xticks(X)

plt.show()
```



<br>
<br>

## 9. 실전문제로 데이터 처리 과정 / 그래프 그리기 익히기


### ex1)

역대 월드컵의 경기당 득점 수<br>

- ax[1].grid(True) # 격자 추가
- data frame의 새로운 열 만들기

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

world_cups = pd.read_csv('WorldCups.csv')
world_cups = world_cups[["Year","GoalsScored","MatchesPlayed"]]
world_cups['GoalsPerMatch'] = world_cups["GoalsScored"] / world_cups["MatchesPlayed"]

fig, ax = plt.subplots(2,1,figsize=(4,8))

ax[0].plot(world_cups["Year"], world_cups["MatchesPlayed"], marker="o",color='blue',label="matches")

ax[0].bar(world_cups["Year"], world_cups["GoalsScored"], color="0.5",label="goals")

ax[0].legend(loc="upper left")
ax[1].grid(True) # 격자 추가
ax[1].plot(world_cups["Year"], world_cups['GoalsPerMatch'], marker="o", color='r',label='goal_per_matches')

ax[1].legend(loc="lower left")
plt.show()
```

### ex2)

역대 월드컵의 국가별 득점 수<br>

- 과정을 중심으로 볼 것

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# 데이터 전처리
#----------------------------------------------------------------------------------------------------------------------------
world_cups_matches = pd.read_csv('WorldCupMatches.csv')
world_cups_matches = world_cups_matches.replace('Germany FR','Germany').replace('C�te d’Ivoire','Côte d’Ivoire').replace\
    ('rn">Bosnia and Herzegovina','Bosnia and Herzegovina').replace('rn">Serbia and Montenegro','Serbia and Montenegro').replace\
    ('rn">Trinidad and Tobago','Trinidad and Tobago').replace('rn">United Arab Emirates','United Arab Emirates').replace\
    ('Soviet Union','Russia').replace('rn">Republic of Ireland','Republic of Ireland')

world_cups_matches = world_cups_matches.drop_duplicates()
#------------------------------------------------------------------------------------------------------------------------------

# 데이터 알맞게 고치기
#------------------------------------------------------------------------------------------------
home = world_cups_matches.groupby(['Home Team Name']).aggregate({'Home Team Goals':np.sum})

away = world_cups_matches.groupby(['Away Team Name']).aggregate({'Away Team Goals':np.sum})

goal_per_country = pd.concat([home,away], axis=1, sort=True).fillna(0)
# concat: away와 home 합치기 (결측값을 제거하기 위해 fillna 함수를 적용합니다.)
# axis=0: 위+아래로 합치기, axis=1: 왼쪽+오른쪽 합치기
goal_per_country['Goal'] = goal_per_country['Home Team Goals'] + goal_per_country['Away Team Goals']
goal_per_country = goal_per_country['Goal'].sort_values(ascending=False) # 값을 가져와서 sort
goal_per_country = goal_per_country.astype(int) # 정수로 변환
#----------------------------------------------------------------------------------------------------

# 그래프 그리기
#-------------------------------------------------------------------------------------------------
goal_per_country = goal_per_country.iloc[:10]
x = goal_per_country.index
y = goal_per_country.values
fig, ax = plt.subplots()
ax.bar(x, y, width=0.5)
plt.xticks(x, rotation=30) # x의 나라 이름들을 30도 기울여준다.
plt.tight_layout() # 그래프가 짤리지 않게 위치조정
plt.show()
#-----------------------------------------------------------------------------------------------
```


### ex3)
2014 월드컵 다득점 국가 순위<br>

- 과정을 중심으로 볼 것

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# 데이터 전처리
#----------------------------------------------------------------------------------------------------------------------------
world_cups_matches = pd.read_csv('WorldCupMatches.csv')
world_cups_matches = world_cups_matches.replace('Germany FR','Germany').replace('C�te d’Ivoire','Côte d’Ivoire').replace\
    ('rn">Bosnia and Herzegovina','Bosnia and Herzegovina').replace('rn">Serbia and Montenegro','Serbia and Montenegro').replace\
    ('rn">Trinidad and Tobago','Trinidad and Tobago').replace('rn">United Arab Emirates','United Arab Emirates').replace\
    ('Soviet Union','Russia').replace('rn">Republic of Ireland','Republic of Ireland')

world_cups_matches = world_cups_matches.drop_duplicates()
#------------------------------------------------------------------------------------------------------------------------------

# 데이터 알맞게 가공
#-----------------------------------------------------------------------------------
world_cups_matches = world_cups_matches[world_cups_matches['Year'] == 2014]

home = world_cups_matches.groupby('Home Team Name')['Home Team Goals'].sum()

away = world_cups_matches.groupby('Away Team Name')['Away Team Goals'].sum()

goals = pd.concat([home,away], axis=1).fillna(0)
goals['Goals'] = goals['Home Team Goals'] + goals['Away Team Goals']
goals = goals['Goals'].sort_values(ascending=False)
goals = goals.astype(int)
#-----------------------------------------------------------------------------------
goals.plot(x= goals.index, y= goals.values, kind="bar", figsize=(12,12),fontsize=14)
plt.tight_layout()

# fig, ax = plt.subplots()
# ax.bar(team_goal_2014.index, team_goal_2014.values)
# plt.xticks(rotation = 90)
# plt.tight_layout()

plt.show()
```

### ex4)
월드컵 4강 이상 성적 집계하기<br>

- 과정과 그래프를 그리는 2가지 방법을 중심으로 보기

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

world_cups = pd.read_csv("WorldCups.csv")
winner = world_cups['Winner']
runners = world_cups['Runners-Up']
third = world_cups['Third']
fourth = world_cups['Fourth']

# value_counts 함수를 이용해 각 시리즈 데이터에 저장된 값을 세어주고 내림차순으로 저장합니다.
winner_count = pd.Series(winner.value_counts())
runners_up_count = pd.Series(runners.value_counts())
third_count = pd.Series(third.value_counts())
fourth_count = pd.Series(fourth.value_counts())

result = pd.concat([winner_count, runners_up_count,third_count,fourth_count], axis=1).fillna(0)
result = result.astype(int)
print(result)
x = np.array(list(range(0,len(result))))


result.plot(y=["Winner", "Runners-Up", "Third", "Fourth"], kind= "bar", color=['gold','silver','brown','black'], figsize=(15,12),
            fontsize=10, width=0.8, align='center') # result 데이터에 직접 plot 함수를 호출 함.
            
            
'''
# 하나하나 설정해 줌
fig, ax = plt.subplots()
plt.xticks(x, result.index, rotation=90)
plt.tight_layout()

ax.bar(x - 0.3, result['Winner'], color='gold', width=0.2, label='Winner')
ax.bar(x - 0.1, result['Runners-Up'], color='silver', width = 0.2, label='Runners_up')
ax.bar(x + 0.1, result['Third'],      color = 'brown',  width = 0.2, label = 'Third')
ax.bar(x + 0.3, result['Fourth'],     color = 'black',  width = 0.2, label = 'Fourth')
'''


plt.show()
```






















































