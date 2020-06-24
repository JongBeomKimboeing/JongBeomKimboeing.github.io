---
layout: post
title: 데이터 분석(Pandas)
description: "데이터 분석(Pandas)"
modified: 2020-06-24
tags: [Pandas]
categories: [Pandas]
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
# 인덱싱을 이용하여 데이터 수정
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

아래 코드는 pandas 연산/함수를 확인하기 위한 data frame을 생성한 코드이다.

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


### 3) Series 연산 (+ - * / 가능)
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

### 4) dataframe 연산 (+ - * / 가능)
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

### 5) dataframe에서도 numpy array에서 사용했던 집계함수 사용이 가능하다.

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





















































































































































