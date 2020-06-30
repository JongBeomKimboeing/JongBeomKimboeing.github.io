---
layout: post
title: 데이터 분석(Pandas)
description: "데이터 분석(Pandas)"
modified: 2020-06-24
tags: [Data Analysis]
categories: [Data Analysis]
---

# Pandas

<pre>
pandas: 구조화된 데이터를 효과적으로 처리하고 저장할 수 있는 파이썬 라이브러리이다.
        Array 계산에 특화된 numpy를 기반으로 만들어져서 다양한 기능들을 제공한다.
</pre>

## 1. Series 데이터형
- Series: numpy array가 보강된 형태로 data와 index를 가지고 있다.

```python
import pandas as pd

data = pd.Series([1,2,3,4])
print(data)
'''
출력결과:
0    1
1    2
2    3
3    4
dtype: int64

앞이 index, 뒤가 data
dtype -> 자료형
'''
```

### 1) series형 데이터의 가장 큰 특징
index를 가지고 있고 index로 접근 가능하다. (딕셔너리와 아주 비슷하다.)

```python
data = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(data)

'''
출력결과:
a    1
b    2
c    3
d    4
dtype: int64
'''

print(data['b']) # result: 2
```

### 2) series 데이터는 딕셔너리로 만들 수 있다.

- dictionary의 키 값은 series의 index로, dictionary의 data 값은 data 값으로 들어간다.
- series.values를 하면 numpu array가 나온다.

```python
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}

population = pd.Series(population_dict)
print(population)

'''
출력결과:

korea      5180
japan     12718
china    141500
usa       32676
dtype: int64
'''

#series.values를 하면 numpu array가 나온다.
print(population.values) # [  5180  12718 141500  32676]
print(type(population.values)) # <class 'numpy.ndarray'>
```
<br>
<br>

## 2. Data Frame
- data frame: 여러 개의 Series가 모여서 행과 열을 이룬 데이터

```python
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)


gdp_dict = {
    'korea':169320000,
    'japan':516700000,
    'china':1409250000,
    'usa':2041280000,
}
#gdp series 데이터
gdp = pd.Series(gdp_dict)


country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
print(country)

'''
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000
'''

```

### 1) Data Frame 속성 확인

- index 확인: country.index
- column 확인: country.columns
- 특정 열 가져오기: country['열 이름']

```python
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)


gdp_dict = {
    'korea':169320000,
    'japan':516700000,
    'china':1409250000,
    'usa':2041280000,
}
#gdp series 데이터
gdp = pd.Series(gdp_dict)


country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
print(country)

'''
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000
'''

#---------------------------------------------------------
#index 확인
print(country.index)
# Index(['korea', 'japan', 'china', 'usa'], dtype='object')

#column 확인
print(country.columns)
# Index(['population', 'gdp'], dtype='object')
# dtype은 문자열이 들어가서 object이다.

print(country['gdp']) # -> series 데이터로 나온다.
'''
korea     169320000
japan     516700000
china    1409250000
usa      2041280000
Name: gdp, dtype: int64
'''
print(type(country['gdp']))
# <class 'pandas.core.series.Series'>

#----------------------------------------------------------

```

### 2) Data frame Series 연산 (뒤에 더 자세히)
- series는 numpy의 보강된 형태이기 때문에, series를 numpy array처럼 연산자를 쓸 수 있다.

```python
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)


gdp_dict = {
    'korea':169320000,
    'japan':516700000,
    'china':1409250000,
    'usa':2041280000,
}
#gdp series 데이터
gdp = pd.Series(gdp_dict)


country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
print(country)

'''
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000
'''

#----------------------------------------------------

gdp_per_capita = country['gdp'] / country['population']
country['gdp_per_capita'] = gdp_per_capita
print(country)
'''
       population         gdp  gdp_per_capita
korea        5180   169320000    32687.258687
japan       12718   516700000    40627.457147
china      141500  1409250000     9959.363958
usa         32676  2041280000    62470.314604
'''

#---------------------------------------------------
```

### 3) 저장과 불러오기

```python
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)


gdp_dict = {
    'korea':169320000,
    'japan':516700000,
    'china':1409250000,
    'usa':2041280000,
}
#gdp series 데이터
gdp = pd.Series(gdp_dict)


country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
print(country)

'''
       population         gdp
korea        5180   169320000
japan       12718   516700000
china      141500  1409250000
usa         32676  2041280000
'''

# 1. 저장
#-----------------------------------
# C:\Users\harry\PycharmProjects\dataAnalysis에 저장됨
country.to_csv("./country.csv")
country.to_excel("country.xlsx")
#----------------------------------


# 2. 불러오기(읽어들이기)
#------------------------------------
#-> dataframe 형태로 읽어들임
country = pd.read_csv("./country.csv")
countrye = pd.read_excel("country.xlsx")
print(country)
print(countrye)
'''
  Unnamed: 0  population         gdp  gdp_per_capita
0      korea        5180   169320000    32687.258687
1      japan       12718   516700000    40627.457147
2      china      141500  1409250000     9959.363958
3        usa       32676  2041280000    62470.314604


  Unnamed: 0  population         gdp  gdp_per_capita
0      korea        5180   169320000    32687.258687
1      japan       12718   516700000    40627.457147
2      china      141500  1409250000     9959.363958
3        usa       32676  2041280000    62470.314604

Process finished with exit code 0

'''
#------------------------------------
```

<br>
<br>

## 3. indexing / slicing

### 1) indexing

#### 1)) loc: 명시적인 인덱스를 참조하는 인덱싱/슬라이싱

- dataframe.loc['인덱스 이름']
- country.loc['korea':'japan', :'population'] -> 이와같이 명시적으로 슬라이싱도 가능
  (index는 korea에서 japan까지(loc indexing에서는 japan 포함)
  (columns는 population까지 (loc indexing에서는 population 포함))


```python

import numpy as np
import pandas as pd

# 두 개의 시리즈 데이터가 있습니다.
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}

population = pd.Series(population_dict)
gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)

country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
gdp_per_capita = country['gdp'] / country['population']
country['gdp per capita'] = gdp_per_capita
print(country)

'''
       population         gdp  gdp per capita
korea        5180   169320000    32687.258687
japan       12718   516700000    40627.457147
china      141500  1409250000     9959.363958
usa         32676  2041280000    62470.314604
'''

print(country.loc['china'])
'''
출력 결과: 

population        1.415000e+05
gdp               1.409250e+09
gdp per capita    9.959364e+03
Name: china, dtype: float64
'''
#-------------------------------------------------------------
print(country.loc['korea':'japan', :'population']) # index는 korea에서 japan까지(loc indexing에서는 japan 포함)
                                                   # columns는 population까지 (loc indexing에서는 population 포함)
'''
       population
korea        5180
japan       12718
'''
#--------------------------------------------------------------
'''
       population         gdp  gdp per capita
korea        5180   169320000    32687.258687
japan       12718   516700000    40627.457147
china      141500  1409250000     9959.363958
usa         32676  2041280000    62470.314604
'''
```


#### 1)) iloc: 파이썬 스타일의 정수 인덱스 인덱싱/슬라이싱
- iloc는 앞에서부터 숫자를 매겨 암묵적으로 인덱싱을 가지고 있어 참조를 가능하게한다.

```python
import numpy as np
import pandas as pd

# 두 개의 시리즈 데이터가 있습니다.
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}

population = pd.Series(population_dict)
gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)

country = pd.DataFrame({
    'population': population,
    'gdp': gdp
})
gdp_per_capita = country['gdp'] / country['population']
country['gdp per capita'] = gdp_per_capita
print(country)

'''
       population         gdp  gdp per capita
korea        5180   169320000    32687.258687
japan       12718   516700000    40627.457147
china      141500  1409250000     9959.363958
usa         32676  2041280000    62470.314604
'''

#-----------------------------------------------------------
print(country.iloc[0]) # korea의 데이터를 가져온다.
'''
population        5.180000e+03
gdp               1.693200e+08
gdp per capita    3.268726e+04
Name: korea, dtype: float64
'''

print(country.iloc[1:3, :2])  #index가 1~2 값(japan, china)이고, columns가 0~1 값인 데이터
'''
       population         gdp
japan       12718   516700000
china      141500  1409250000
'''
#----------------------------------------------------------
```

<br>
<br>

## 4. dataframe 새 데이터 추가/수정

- data frame 만들기

```python
dataframe = pd.DataFrame(columns=['이름', '나이', '주소'])
print(dataframe)
print('\n')
'''
Empty DataFrame
Columns: [이름, 나이, 주소]
Index: []
'''
```

- 리스트로 데이터 추가

```python
# 리스트로 데이터 추가
dataframe.loc[0] = ['임원균', '26', '서울']
print(dataframe)
print('\n')
'''
    이름  나이  주소
0  임원균  26  서울
'''
```

- 딕셔너리로 데이터 추가

```python
#딕셔너리로 데이터 추가
dataframe.loc[1] = {'이름':'철수', '나이':'25', '주소':'인천'}
print(dataframe)
print('\n')
'''
    이름  나이  주소
0  임원균  26  서울
1   철수  25  인천
'''
```

- 인덱싱을 이용하여 데이터 수정

```python

#인덱싱을 이용하여 데이터 수정

dataframe.loc[1, '이름'] = '영희'
print(dataframe)
print('\n')

'''
    이름  나이  주소
0  임원균  26  서울
1   영희  25  인천
'''

```

- 새 칼럼 추가

```python
dataframe['전화번호'] = np.nan # 컬럼을 추가하고 값은 비워둔다.
print(dataframe)
print('\n')
'''
    이름  나이  주소  전화번호
0  임원균  26  서울   NaN
1   영희  25  인천   NaN
'''
```

- 인덱싱으로 데이터 추가

```python
dataframe.loc[0,'전화번호'] = '01012341234' # 인덱싱으로 데이터 추가
print(dataframe)
print('\n')
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천          NaN
'''
```

### 추가

#### len(dataframe)

```python
print(len(dataframe)) # 2 -> 0과 1을 갖고 있으므로, 결과가 2가 나온다.
```

#### data frame column 선택하기

- column 이름이 하나만 있다면 Series
- list로 들어가 있다면 DataFrame

```python
print(dataframe['이름']) # 이름에 해당하는 column을 가져온다.(column이 하나이므로, Series데이터이다.)
'''
0    임원균
1     영희
Name: 이름, dtype: object
'''
print(dataframe[["이름","주소","나이"]]) # 이름에 해당하는 column을 가져온다.(column이 여러개(list형)이므로, column 데이터이다.)
'''
    이름  주소  나이
0  임원균  서울  26
1   영희  인천  25
'''
```


### dataframe 새 데이터 추가/수정 전체 코드

```python
dataframe = pd.DataFrame(columns=['이름', '나이', '주소'])
print(dataframe)
print('\n')
'''
Empty DataFrame
Columns: [이름, 나이, 주소]
Index: []
'''
# 리스트로 데이터 추가
dataframe.loc[0] = ['임원균', '26', '서울']
print(dataframe)
print('\n')
'''
    이름  나이  주소
0  임원균  26  서울
'''
#딕셔너리로 데이터 추가
dataframe.loc[1] = {'이름':'철수', '나이':'25', '주소':'인천'}
print(dataframe)
print('\n')
'''
    이름  나이  주소
0  임원균  26  서울
1   철수  25  인천
'''
# 인덱싱을 이용하여 데이터 수정
dataframe.loc[1, '이름'] = '영희'
print(dataframe)
print('\n')
'''
    이름  나이  주소
0  임원균  26  서울
1   영희  25  인천
'''

#dataframe 새 컬럼 추가

dataframe['전화번호'] = np.nan # 컬럼을 추가하고 값은 비워둔다.
print(dataframe)
print('\n')
'''
    이름  나이  주소  전화번호
0  임원균  26  서울   NaN
1   영희  25  인천   NaN
'''
dataframe.loc[0,'전화번호'] = '01012341234' # 인덱싱으로 데이터 추가
print(dataframe)
print('\n')
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천          NaN
'''
print(len(dataframe)) # 2 -> 0과 1을 갖고 있으므로, 결과가 2가 나온다.

# column 선택하기
# column 이름이 하나만 있다면 Series
# list로 들어가 있다면 DataFrame

print(dataframe['이름']) # 이름에 해당하는 column을 가져온다.(column이 하나이므로, Series데이터이다.)
'''
0    임원균
1     영희
Name: 이름, dtype: object
'''

print(dataframe[["이름","주소","나이"]]) # 이름에 해당하는 column을 가져온다.(column이 여러개(list형)이므로, column 데이터이다.)
'''
    이름  주소  나이
0  임원균  서울  26
1   영희  인천  25
'''
```

<br>
<br>

## 5. Pandas 연산과 함수

(아래 코드는 pandas 연산/함수를 확인하기 위한 data frame을 생성한 코드이다.)

```python

import numpy as np
import pandas as pd
dataframe = pd.DataFrame(columns=['이름', '나이', '주소'])
dataframe.loc[0] = ['임원균', '26', '서울']
dataframe.loc[1] = {'이름':'철수', '나이':'25', '주소':'인천'}

dataframe.loc[1, '이름'] = '영희'
dataframe['전화번호'] = np.nan 

dataframe.loc[0,'전화번호'] = '01012341234'
print(dataframe)

'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천          NaN
'''
```

### 1) 누락된 데이터 체크
튜토리얼에서 보는 데이터와 달리 현실의 데이터는 누락되어 있는 형태가 많다. <br>
데이터에 nan이나 None이 있으면 비어있다고 판단한다.<br>

<br>

- dataframe.isnull()

데이터가 비어있으면 True, 데이터가 채워있으면 False

```python
print(dataframe.isnull()) # 데이터가 비어있으면 True, 데이터가 채워있으면 False

'''
      이름     나이     주소   전화번호
0  False  False  False  False
1  False  False  False   True
'''
```

- dataframe.notnull()

데이터가 채워있으면 True, 데이터가 비어있으면 False

```python
print(dataframe.notnull()) # 데이터가 채워있으면 True, 데이터가 비어있으면 False
'''
     이름    나이    주소   전화번호
0  True  True  True   True
1  True  True  True  False
'''
```

### 2) 누락된 데이터 삭제하거나 채워주기

#### 1. 누락된 데이터 삭제하기

- dataframe.dropna()

앞에서 보면, 영희의 전화전호가 비어있는데, 이러한 빈 데이터를 지우기 위해 dropna()를 이용한다.

```python
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천          NaN
'''

print(dataframe.dropna()) # 앞에서 보면, 영희의 전화전호가 비어있는데, 이러한 빈 데이터를 지우기 위해 dropna()를 이용한다.
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
'''

```

#### 2. 누락된 데이터 채워주기

- dataframe['전화번호'].fillna('전화번호 없음')

빈 데이터를 채워주기 위해 fillna를 사용한다.

```python
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천          NaN
'''

dataframe['전화번호'] = dataframe['전화번호'].fillna('전화번호 없음')
# 앞에서 보면, 영희의 전화전호가 비어있는데, 이러한 빈 데이터를 채워주기 위해 fillna를 사용한다.
print(dataframe)
'''
    이름  나이  주소         전화번호
0  임원균  26  서울  01012341234
1   영희  25  인천      전화번호 없음
'''
```


### 3. Series 연산 (+ - * / 가능)
numpy array에서 사용했던 연산자들을 활용할 수 있다.<br>

**아래 코드에서 보는바와 같이 numpy array에서 사용했던 연산자들을 그대로 사용 가능하다.**

```python
A = pd.Series([2,4,6], index=[0,1,2])
print(A)
print('\n')
'''
0    2
1    4
2    6
dtype: int64
'''
B = pd.Series([1,3,5], index=[1,2,3])
print(B)
print('\n')
'''
1    1
2    3
3    5
dtype: int64
'''

print(A + B) # 서로 인덱스가 같은 값끼리 구한다. 만약, 같은 index의 값이 없으면, 계산을 못 하고 결과가 Nan이 된다.
'''
0    NaN
1    5.0
2    9.0
3    NaN
dtype: float64
'''

print(A.add(B, fill_value=0)) # fill_value=0:인덱스의 값이 비어있으면 0으로 채워준다.
#그러므로, 아래와 같은 연산결과가 나온다.
'''
0    2.0
1    5.0
2    9.0
3    5.0
dtype: float64
'''
```

### 4. dataframe 연산 (+ - * / 가능)
numpy array에서 사용했던 연산자들을 활용할 수 있다.<br>

**아래 코드에서 보는바와 같이 numpy array에서 사용했던 연산자들을 그대로 사용 가능하다.**

```python
a = pd.DataFrame(np.random.randint(0,10,(2,2)),columns=list("AB"))
print(a)
print('\n')
'''
   A  B
0  1  2
1  1  9
'''
b = pd.DataFrame(np.random.randint(0,10,(3,3)), columns=list("BAC"))
print(b)
print('\n')
'''
   B  A  C
0  1  7  4
1  5  5  2
2  6  7  7
'''
print(a+b)
'''
     A     B   C
0  8.0   3.0 NaN
1  6.0  14.0 NaN
2  NaN   NaN NaN
'''
print('\n')

print(a.add(b, fill_value=0)) # 없는 data에 대해 0으로 채워줘서 계산함
'''
      A     B    C
0  13.0   9.0  0.0
1  13.0  10.0  8.0
2   5.0   8.0  4.0
'''

```

### 5. dataframe에서도 numpy array에서 사용했던 집계함수 사용이 가능하다.

```python
data = {
    'A': [i+5 for i in range(3)],
    'B': [i**2 for i in range(3)]
}
df = pd.DataFrame(data)
print(df)
print('\n')
'''
   A  B
0  5  0
1  6  1
2  7  4
'''
print(df['A'].sum()) #A Series 데이터를 합한다
print('\n')
# 18
print(df.sum()) # A series 데이터와 B series 데이터를 더한다.
print('\n')
'''
A    18
B     5
dtype: int64
'''
print(df.mean())
print('\n')
'''
A    6.000000
B    1.666667
dtype: float64
'''

```

<br>
<br>

## 6. dataframe 정렬하기(sort)

(아래 코드는 dataframe 정렬을 확인하기 위한 data frame을 생성한 코드이다.)

```python
df = pd.DataFrame({
    'col1':[2, 1, 9, 8, 7, 4],
    'col2':['A','A','B',np.nan,'D','C'],
    'col3':[0, 1, 9, 4, 2, 3]
})

print(df)
'''
   col1 col2  col3
0     2    A     0
1     1    A     1
2     9    B     9
3     8  NaN     4
4     7    D     2
5     4    C     3
'''
```

<br>

### 1) sort_values()
-> 데이터들을 값으로 재정렬

```python
print(df.sort_values('col1')) # 지정한 column의 값을 기준으로 정렬이 된다.(오름차순)
'''
   col1 col2  col3
1     1    A     1
0     2    A     0
5     4    C     3
4     7    D     2
3     8  NaN     4
2     9    B     9
'''

print(df.sort_values('col1', ascending=False)) # 지정한 column의 값을 기준으로 정렬이 된다.(내림차순)
'''
   col1 col2  col3
2     9    B     9
3     8  NaN     4
4     7    D     2
5     4    C     3
0     2    A     0
1     1    A     1
'''
print(df.sort_values(['col2','col1'])) #col2를 기준으로 정렬 후, col2 중에 값이 같은 것이 있다면 col1을 기준으로 정렬한다.(오름차순)
'''
   col1 col2  col3
1     1    A     1       -> A 값이 같음 col1을 기준으로 정렬 그러므로 1 
0     2    A     0       -> A 값이 같음col1을 기준으로 정렬 그러므로 2
2     9    B     9
5     4    C     3
4     7    D     2
3     8  NaN     4
'''
```

### 2) sort_value() 예제

```python
import numpy as np
import pandas as pd

print("DataFrame: ")
df = pd.DataFrame({
    'col1' : [2, 1, 9, 8, 7, 4],
    'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col3': [0, 1, 9, 4, 2, 3],
})
print(df, "\n")


# 정렬 코드 입력해보기
# Q1. col1을 기준으로 오름차순으로 정렬하기.
print(df.sort_values('col1'))


# Q2. col2를 기준으로 내림차순으로 정렬하기.
print(df.sort_values('col2', ascending=False))


# Q3. col2를 기준으로 오름차순으로, col1를 기준으로 내림차순으로 정렬하기.

print(df.sort_values(['col2', 'col1'], ascending=[True,False]))
'''
ascending 매개변수에 리스트 형식으로 True 및 False를 각각 대입해주면 col2에는 오름차순,
col1에는 내림차순으로 정렬 기준이 각각 적용됩니다.
'''
```

<br>
<br>

## 7. 조건으로 검색하기

### 1) masking 연산
- numpy array와 마찬가지로 masking 연산이 가능하다.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(5,2), columns=["A", "B"])
print(df)

'''
          A         B
0  0.755695  0.077526
1  0.299311  0.298824
2  0.049641  0.188899
3  0.094146  0.888122
4  0.689220  0.919613
'''

#-------------------------
print(df["A"] < 0.5)
print('\n')

'''
0    False
1     True
2     True
3     True
4    False
Name: A, dtype: bool
'''
print('\n')

```


### 2) 조건에 맞는 dataframe row를 추출 가능하다.

```python
import numpy as np
import pandas as pd

dd = pd.DataFrame(np.random.rand(5,2), columns=["A", "B"])

print((dd["A"] < 0.5))
print('\n')
'''
0     True
1    False
2     True
3    False
4     True
Name: A, dtype: bool
'''
print((dd["B"] > 0.3))
print('\n')

'''
0    False
1    False
2     True
3    False
4     True
Name: B, dtype: bool
'''
```

### 3) 논리 연산자 이용

```python
import numpy as np
import pandas as pd

dd = pd.DataFrame(np.random.rand(5,2), columns=["A", "B"])


print(dd[(dd["A"] < 0.5) & (dd["B"] > 0.3)]) # A, B 모두 true인 값들을 찾아내서 추출한다.
print('\n')
'''
          A         B
2  0.133516  0.703919
4  0.068774  0.731063
'''
```

### 4) query를 통해 조건에 따라 추출할 수 있다.

```python
import numpy as np
import pandas as pd

dd = pd.DataFrame(np.random.rand(5,2), columns=["A", "B"])

print(dd.query("A < 0.5 and B>0.3")) #query를 통해 조건에 따라 추출할 수 있다.
print('\n')
'''
          A         B
2  0.133516  0.703919
4  0.068774  0.731063
'''
```

### 5) 문자열 비교로 검색하기

```python
Animal = ["Dog","Cat","Cat","Pig","Cat"]
Name = ["Happy","Sam", "Toby","Mini","Rocky"]
df = pd.DataFrame({
    'Animal':Animal,
    'Name':Name
})
print(df)
'''
  Animal   Name
0    Dog  Happy
1    Cat    Sam
2    Cat   Toby
3    Pig   Mini
4    Cat  Rocky

'''
```

- contains를 이용하여 Cat을 포함하면 true 아니면 false

```python
print(df["Animal"].str.contains("Cat")) # contains를 이용하여 Cat을 포함하면 true 아니면 false
'''
0    False
1     True
2     True
3    False
4     True
Name: Animal, dtype: bool
'''
```

- 문자열을 바로 비교한다.

```python
print(df["Animal"] == "Cat") #문자열을 바로 비교해도 됨
'''
0    False
1     True
2     True
3    False
4     True
Name: Animal, dtype: bool
'''
```

- match를 이용하여 검색해낼 수 있다.

(match 안에 정규표현식이라고 하는 문자열을 다루기 편리한 도구를 사용할 수 있다.)

```python
print(df.Animal.str.match("Cat")) #match를 이용하여 Cat을 검색해낼 수 있다.
'''
0    False
1     True
2     True
3    False
4     True
Name: Animal, dtype: bool
'''
# match 안에 정규표현식이라고 하는 문자열을 다루기 편리한 도구를 사용할 수 있다.
```

<br>
<br>

## 8. 함수로 데이터 처리하기

- apply를 통해서 함수로 데이터를 다룰 수 있다.

```python

df = pd.DataFrame(np.arange(5), columns=["Num"])

def square(x):
    return x**2

print(df["Num"].apply(square)) # apply 함수에 인자를 바로 함수이름을 넣어준다. (series형)
'''
0     0
1     1
2     4
3     9
4    16
Name: Num, dtype: int64
'''
df["square"] = df["Num"].apply(square)
print(df)
'''
   Num  square
0    0       0
1    1       1
2    2       4
3    3       9
4    4      16
'''
```

- lamda 표현식을 사용하여 간단하게 표현 할 수 있다.

```python
df = pd.DataFrame(np.arange(5), columns=["Num"])

df["square"] = df.Num.apply(lambda x: x**2) # lamda 표현식을 사용하여 간단하게 표현 할 수 있다.
# lambda: 입력을 받아주고 바로 값을 return 해준다.
```

- 예시
(apply를 이용하여 문자를 숫자로 변환하기)<br>

<br>

복잡한 연산을 하거나, 데이터를 처리할 때 apply를 통해서 처리를 할 수 있다.

```python
df = pd.DataFrame(columns=["phone"])
df.loc[0] = "010-1234-1235"
df.loc[1] = "공일공-일이삼사-1235"
df.loc[2] = "010.1234.일이삼오"
df.loc[3] = "공1공-1234-1이3오"
df["preprocess_phone"] = ''

print(df)
'''
           phone preprocess_phone
0  010-1234-1235                 
1  공일공-일이삼사-1235                 
2  010.1234.일이삼오                 
3  공1공-1234-1이3오  
'''

def get_preprocess_phone(phone):
    mapping_dict = {
        "공":"0",
        "일":"1",
        "이":"2",
        "삼":"3",
        "사":"4",
        "오":"5",
        "-":"",
        ".":""
    }
    for key, value in mapping_dict.items():
        phone = phone.replace(key, value)
    return phone
df["preprocess_phone"] = df["phone"].apply(get_preprocess_phone)
print(df)
'''
           phone preprocess_phone
0  010-1234-1235      01012341235
1  공일공-일이삼사-1235      01012341235
2  010.1234.일이삼오      01012341235
3  공1공-1234-1이3오      01012341235
'''
# 복잡한 연산을 하거나, 데이터를 처리할 때 apply를 통해서 처리를 할 수 있다.

```

<br>
<br>

## 9. 특정한 data 값을 다른 data 값으로 바꾸기
### replace를 이용한다.
-> replace: apply 기능에서 데이터 값만 대체하고 싶을 때 이용

```python
gender = pd.Series(["Male", "Male", "Female", "Female", "Male"])
df = pd.DataFrame({
    'gender':gender
})
print(df)
'''
0      Male
1      Male
2    Female
3    Female
4      Male
dtype: object
'''
print(df.gender.replace({"Male":0, "Female":1})) #Male은 0으로, Female은 1로 바뀐다.
'''
0    0
1    0
2    1
3    1
4    0
Name: gender, dtype: int64
'''
df["gender"] = df.gender.replace({"Male":0, "Female":1})# Male은 0으로, Female은 1로 바꾸고 이를 적용시킨다.
print(df)
'''
   gender
0       0
1       0
2       1
3       1
4       0
'''
df.gender.replace({"Male":0, "Female":1}, inplace=True) # inplace를 이용하여 Male은 0으로, Female은 1로 바꾸고 이를 적용시킨다.
print(df)
'''
   gender
0       0
1       0
2       1
3       1
4       0
'''
#inplace를 통해 바로 dataframe에 적용시킬 수 있다.
```

<br>
<br>

## 10. 그룹으로 묶기
간단한 집계를 넘어서서 조건부로 집계하고 싶은 경우 이용한다.

- key가 같은 것끼리 그룹으로 묶고 그룹끼리 연산 수행

```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[1, 2, 3, 1, 2, 3],
                   'data2': np.random.randint(0, 6, 6)
                   })
print(df)
'''
  key  data1  data2
0   A      1      0
1   B      2      3
2   C      3      3
3   A      1      2
4   B      2      3
5   C      3      2
'''

df.groupby('key') # key가 같은 것끼리 묶인 채로 가지고만 있는다.
dd =  df.groupby('key').sum() # 그룹이 묶인 상태로 연산 수행 (여기서는 key를 기준으로 합계를 구해달라)
print(dd)
'''
     data1  data2
key              
A        2      2
B        4      6
C        6      5
'''
```

-  여러개로 묶인 경우

(여러개로 묶인 경우 key, data1이 같은 값끼리 묶이고, sum을 하면 data2 값끼리 연산이 된다.)

```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[1, 2, 3, 1, 2, 3],
                   'data2': np.random.randint(0, 6, 6)
                   })
print(df)
'''
  key  data1  data2
0   A      1      0
1   B      2      3
2   C      3      3
3   A      1      2
4   B      2      3
5   C      3      2
'''

dd = df.groupby(['key', 'data1']).sum() # 여러개로 묶인 경우 key, data1이 같은 값끼리 묶이고, sum을 하면 data2 값끼리 연산이 된다.
print(dd)

```

## 11. 그룹으로 묶고 연산하기

### 1) aggregate 연산
- groupby를 통해서 집계를 한번에 계산하는 방법

ex1)
```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[1, 2, 3, 1, 2, 3],
                   'data2': np.random.randint(0, 6, 6)
                   })
print(df)
'''
  key  data1  data2
0   A      1      0
1   B      2      3
2   C      3      3
3   A      1      2
4   B      2      3
5   C      3      2
'''

df = df.groupby('key').aggregate(['min',np.median, max]) #키로 묶인 값들 끼리 [가장작은 값, 중간 값, 가장 큰 값]으로 연산한다.
print(df)

'''
    data1            data2           
      min median max   min median max
key                                  
A       1      1   1     1    1.5   2
B       2      2   2     2    3.0   4
C       3      3   3     0    0.5   1
'''
```

<br>

ex2)<br>
column마다 어떠한 연산을 수행할지도 지정해서 수행할 수 있다.

```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[1, 2, 3, 1, 2, 3],
                   'data2': np.random.randint(0, 6, 6)
                   })
print(df)
'''
  key  data1  data2
0   A      1      0
1   B      2      3
2   C      3      3
3   A      1      2
4   B      2      3
5   C      3      2
'''

df = df.groupby('key').aggregate({'data1':'min', 'data2':np.sum})
print(df)
# data1은 묶인 key 값 중에서 작은 값을 가져온다.
# data2는 묶인 key 값 끼리의 합을 계산한다.
# 이와같이 column마다 어떠한 연산을 수행할지도 지정해서 수행할 수 있다.
#그러므로 집계 연산을 한번에 계산 가능하다.
'''
     data1  data2
key              
A        1      8
B        2      5
C        3      1
'''
```

### 2) filter
**groupby를 통해서 그룹 속성을 기준으로 데이터 필터링**
- filter를 이용하여 함수를 적용시켜 값을 필터링할 수 있다
- 필터 연산은 True, False의 조건을 넣어 true면 data에 담아주고, false이면 data에 담아두지 않고 버린다.



```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[0, 1, 2, 3, 4, 5],
                   'data2': np.random.randint(0, 6, 6)
                   })

'''
  key  data1  data2
0   A      0      4
1   B      1      5
2   C      2      1
3   A      3      5
4   B      4      5
5   C      5      3
'''

def filter_by_mean(x):
    return x['data2'].mean()>3

print(df.groupby('key').mean()) # 'key'로 묶인 상태에서(A,B,C) data1과 data2의 평균을 각각 구한다.
'''
     data1  data2
key              
A      1.5    4.5
B      2.5    5.0
C      3.5    2.0
'''
print(df.groupby('key').filter(filter_by_mean)) #filter를 이용하여 함수를 적용시켜 값을 필터링할 수 있다
# 필터 연산은 True, False의 조건을 넣어 true면 data에 담아주고, false이면 data에 담아두지 않고 버린다.
'''
  key  data1  data2
0   A      0      4
1   B      1      5
3   A      3      5
4   B      4      5
'''

```


### 3) apply
groupby에도 apply 함수가 적용 가능하며, groupby를 통해서 묶인 데이터에 함수 적용

```python
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1':[0, 1, 2, 3, 4, 5],
                   'data2': np.random.randint(0, 6, 6)
                   })

'''
  key  data1  data2
0   A      0      4
1   B      1      5
2   C      2      1
3   A      3      5
4   B      4      5
5   C      5      3
'''

# groupby 된 채로 apply를 적용하여 새로운 데이터를 만들 수 있다.
print(df.groupby('key').apply(lambda x: x.max() - x.min()))
'''
     data1  data2
key              
A        3      1
B        3      0
C        3      2
'''
```

### cf) get_group
groupby로 묶인 데이터에서 key 값으로 데이터를 가져올 수 있다.


<br>
<br>

## 12. 
Index & pivot_table

### 1) MultiIndex
- 인덱스를 계층적으로 만들 수 있다.

```python

df = pd.DataFrame(
    np.random.randn(4,2),
    index=[['A','A','B','B'], [1, 2, 1, 2]],
    columns=['data1','data2']
)

# 'A'와 'A'가 하나로 묶이고, 'B'와 'B'가 하나로 묶인다.
print(df)
'''
        data1     data2
A 1 -0.652179  0.566551
  2  0.670209  1.059119
B 1  0.276043  1.093460
  2  0.827762 -0.358252
'''

```

- 열 인덱스도 계층적으로 만들 수 있다.

```python
df = pd.DataFrame(
    np.random.randn(4,4),
    columns=[["A","A","B","B"],["1","2","1","2"]]
)
print(df)
'''
          A                   B          
          1         2         1         2
0 -0.145184 -0.289167  0.772888 -0.782718
1 -0.484876 -0.044394 -1.039765 -0.838555
2  0.365689  0.208759 -0.395191  0.400229
3  0.560826 -0.162387  0.710259 -2.166227

'''
```

### 1-1) MultiIndex 인덱싱
- 인덱스 탐색의 경우에는 loc, iloc를 사용가능하다.
- column을 가져올 때는 반드시 '[]'를 이용하여 인덱싱을 한다.

```python
'''
          A                   B          
          1         2         1         2
0 -0.145184 -0.289167  0.772888 -0.782718
1 -0.484876 -0.044394 -1.039765 -0.838555
2  0.365689  0.208759 -0.395191  0.400229
3  0.560826 -0.162387  0.710259 -2.166227

'''

print(df["A"]) # A 부분을 가져온다.
'''
          1         2
0 -0.145184 -0.289167
1 -0.484876 -0.044394
2  0.365689  0.208759
3  0.560826 -0.162387

'''
print(df["A"]["1"]) # A의 1 부분을 가져온다.
'''
0   -0.145184
1   -0.484876
2    0.365689
3    0.560826
Name: 1, dtype: float64
'''
```

- index를 가져올 때는 반드시 loc, iloc를 이용해야한다.

```python
df2 = pd.DataFrame(
    np.random.randn(4, 2),
    index=[["A", "A", "B", "B"], ["1", "2", "1", "2"]],
    columns=["data1","data2"]
)

print("DataFrame2")
print(df2, "\n")
'''
        data1     data2
A 1  0.797239  0.883021
  2 -0.821758  0.410599
B 1 -0.335025  0.019638
  2 -0.837276  0.117544 
'''


print(df2.loc["A","1"]) # loc를 이용한 인덱싱
'''
data1    0.797239
data2    0.883021
Name: (A, 1), dtype: float64
'''
```

### 2) pivot_table
데이터에서 필요한 자료만 뽑아서 새롭게 요약, 분석할수 있는 기능이다. (엑셀에서의 피봇 데이블과 같다.)
<pre>
index는 행 인덱스로 들어갈 key
column에 열 인덱스로 라벨링될 값
value에 분석할 데이터가 들어간다.
aggfunc은 값을 어떻게 채울 것인가?
</pre>

```python
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('the_pied_piper_of_hamelin.csv')
    # 파일을 읽어서 코드를 작성해보세요
    # 경로: "./data/the_pied_piper_of_hamelin.csv"
    df = df[df["구분"] == "Child"]
    #mean = df.groupby(["일차","성별"]).aggregate({"나이":np.mean})
    #print(mean)
    result = df.pivot_table(
        index="일차",
        columns="성별",
        values="나이",
        aggfunc= np.mean #값을 어떻게 채울 것인가? -> "일차와 성별별 나이의 평균을 구한다."
    )
    print(result)
    '''
    성별    Female      Male
일차                    
3   9.500000  9.000000
4   9.000000  6.333333
5   8.666667  8.833333
6   9.411765  7.846154
    '''

    for name in df["이름"].unique(): # 이름 데이터를 일차로 나누어지지 않고 모든 아이들의 이름을 보기 위해 unique함수를 이용한다.
        print(name)
if __name__ == "__main__":
    main()
```










































































































