---
layout: post
title: Kaggle workflow
description: " Kaggle workflow"
modified: 2020-08-11
tags: [Kaggle]
categories: [Kaggle]
---

#Kaggle workflow<br>
<br>
Part1: Exploratory Data Analysis(EDA):<br>
<br>
1)Analysis of the features. (특성과 label 이 어떤 관계를 보이는 지 확인한다.)<br>
<br>
2)Finding any relations or trends considering multiple features. ()<br>
<br>
Part2: Feature Engineering and Data Cleaning:<br>
1)Adding any few features.<br>
<br>
2)Removing redundant features.<br>
<br>
3)Converting features into suitable form for modeling.<br>
<br>
Part3: Predictive Modeling<br>
1)Running Basic Algorithms.<br>
<br>
2)Cross Validation.<br>
<br><br>
3)Ensembling.<br>
<br>
4)Important Features Extraction.<br>

```python
# Part1: Exploratory Data Analysis(EDA):

# 1)Analysis of the features. (특성과 label 이 어떤 관계를 보이는 지 확인한다.)

# 1. 데이터 불러오기
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def get_train_test_data():
    train_data = pd.read_csv("titanic/train.csv")
    test_data = pd.read_csv("titanic/test.csv")
    return train_data, test_data

train_raw_data, test_raw_data = get_train_test_data()

# 2. 데이터 head 확인하기
#print(train_raw_data.head())

# 3. 데이터 결측값 확인하기

#print(train_raw_data.isnull().sum())

'''
[5 rows x 12 columns]
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
'''

# 4. 결측값 채우기

def train_preprocess(train_raw_data):
    train_raw_data["Cabin"].fillna('N', inplace=True)
    train_raw_data['Embarked'].fillna('S', inplace=True)
    train_raw_data["Cabin"] = train_raw_data["Cabin"].str[:1]
    return train_raw_data

def test_preprocess(test_raw_data):
    test_raw_data["Cabin"].fillna('N', inplace=True)
    test_raw_data['Embarked'].fillna('S', inplace=True)
    test_raw_data["Cabin"] = test_raw_data["Cabin"].str[:1]
    return test_raw_data

train_data = train_preprocess(train_raw_data)
test_data = test_preprocess(test_raw_data)

# 5. label 살펴보기
def how_label(train_data):
    fig, ax = plt.subplots(1,2, figsize=(15,8))
    train_data['Survived'].value_counts().plot.pie(explode=[0,0], autopct='%1.1f%%', ax = ax[0])
    # explode=[0,0] -> 각 특성별 차트를 얼마나 띄울 것인지 (사실 중요하게 필요하지 않음)
    # autopct='%1.1f%%' -> 중요함! -> 퍼센트를 계산해서 그래프에 찍어줌
    # ax도 지정 가능
    ax[0].set_title("Survived")
    sns.countplot(x='Survived', data=train_data, ax=ax[1]) # 개수를 세서 그래프 그릴 수 있다.
    plt.show()

#how_label(train_data)


# 6. label 과 각각의 feature 사이의 관계 알아보기

# 1)) Categorical Feature 와 Ordinal Feature 는 주로 막대 그래프나 factorplot을 이용한다.

# 1) sex 와 label
def sex_survived_plot(train_data):
    fig, ax = plt.subplots(1,2, figsize=(10,5)) # 1 행 2열
    gender_graph1 = train_data.groupby(['Sex', 'Survived'])['Survived'].count()
    gender_graph = pd.crosstab(train_data.Sex, train_data.Survived, margins=True)
    # sex와 survive 관계
    sns.barplot(x='Sex', y='Survived', data=train_data, ax=ax[0])  # 0 이 female, 1이 male
    # male에서 survive, female에서 survive 관계
    sns.countplot(x='Sex', hue='Survived', data=train_data, ax=ax[1])
    plt.show()
#sex_survived_plot(train_data) # 상관관계가 있다!


# 2) 좌석 class 별 survive

def class_survived_plot(train_data):
    # 클래스가 복수개 일 때 cross_tab 사용
    cs = pd.crosstab(train_data.Pclass, train_data.Survived, margins=True) # margin 은 총 개수를 출력해준다.
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    train_data['Pclass'].value_counts().plot.bar(ax=ax[0])
    sns.countplot(x='Pclass', hue='Survived', data=train_data, ax=ax[1])
    plt.show()

#class_survived_plot(train_data) # 1등석 사람들이 많이 살아 남은 것을 볼 수 있다.


# 3) 성별과 class 별 survive를 조사해 본다.
# 여러가지 클래스를 섞어서 조사해보기도 한다.

def sex_class_survive(train_data):
    sex_class_df = pd.crosstab([train_data.Sex, train_data.Pclass], train_data.Survived, margins=True)
    print(sex_class_df)
    # 여러가지 클래스를 섞어서 사용할 경우 factorplot 을 사용하는 게 좋다.
    sns.factorplot('Pclass', 'Survived', hue='Sex', data=train_data)
    plt.show()

#sex_class_survive(train_data) # Pclass와 성별이 생존에 많은 영향을 미친다는 것을 알 수 있다.



# 2))  Continous Feature 알아보기

# 1) age 별 survival 알아보기
#print('Oldest Passenger was of:',train_data['Age'].max(),'Years')
#print('Youngest Passenger was of:',train_data['Age'].min(),'Years')
#print('Average Age on the ship:',train_data['Age'].mean(),'Years')

# 연속된 데이터이므로 violinplot 을 이용해보자.

def age_survived(train_data):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train_data, split=True, ax=ax[0])
    # split 으로 survived/non survived를 나눌 수 있다.
    sns.violinplot(x='Sex', y='Age', hue='Survived', data=train_data, split=True, ax=ax[1])
    plt.show()
#age_survived(train_data)


# age 에 nan 값이 있는데 이를 채워줘야 한다.
# 이름 데이터를 이용하여 더 정확한 값을 채원준다.

def fill_age_with_name(train_data):
    print(train_data['Name'])
    train_data['initial'] = 0
    test_data['initial'] = 0
    for i in train_data:
        train_data['initial'] = train_data.Name.str.extract('([A-Za-z]+)\.') # . 앞에 글자 추출하기
    for i in test_data:
        test_data['initial'] = test_data.Name.str.extract('([A-Za-z]+)\.') # . 앞에 글자 추출하기
    print(train_data['initial'])
    # 잘못 된 스펠링 찾기
    initial_sex = pd.crosstab(train_data.initial, train_data.Sex)
    print(initial_sex)
    train_data['initial'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Mlle','Mme','Ms','Rev','Sir'],
                                  ['Mr', 'Other','Mrs','Mr', 'Mr','Other','Mrs', 'Mr','Miss','Miss','Miss','Other','Mr'], inplace=True)
    test_data['initial'].replace(
        ['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Ms', 'Rev', 'Sir','Dona'],
        ['Mr', 'Other', 'Mrs', 'Mr', 'Mr', 'Other', 'Mrs', 'Mr', 'Miss', 'Miss', 'Miss', 'Other', 'Mr','Mr'], inplace=True)
    print(train_data.groupby('initial')['Age'].mean()['Mr'])

    # 채워주기
    for i in ['Master', 'Miss', 'Mr', 'Mrs','Other']:
        train_data.loc[(train_data.Age.isnull()) & (train_data.initial == i), 'Age'] = train_data.groupby('initial')['Age'].mean()[i]

    for i in ['Master', 'Miss', 'Mr', 'Mrs','Other']:
        test_data.loc[(test_data.Age.isnull()) & (test_data.initial == i), 'Age'] = test_data.groupby('initial')['Age'].mean()[i]

    print(train_data['Age'].isnull().any()) # 다 채워졌는 지 확인
    '''
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    train_data[train_data['Survived'] == 0].Age.plot.hist(ax=ax[0], bins=20,edgecolor='black', color='red')
    ax[0].set_title('Survived = 0')
    train_data[train_data['Survived'] == 1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black', color='green')
    ax[1].set_title('Survived = 1')
    plt.show()

    fig, ax = plt.subplots()
    sns.violinplot(x='Sex', y='Age', hue='Survived', data=train_data)
    plt.show()
    '''

fill_age_with_name(train_data)


# 탄 곳 (embarked)

def embarked_survived(train_data):
    print(pd.crosstab([train_data.Embarked, train_data.Pclass], [train_data.Sex, train_data.Survived], margins=True))

    # 탄 곳에 따라 다른 생존 확률
    # Categorical Value (여러가지 카테고리가 있는 경우) 인 경우 barplot/factorplot 을 이용하여 시각화한다.
    sns.barplot(x='Embarked', y='Survived', data=train_data)
    plt.show()
    sns.factorplot(x='Embarked', y='Survived', data=train_data)
    plt.show()

    fig, ax = plt.subplots(2,2,figsize=(15,10))
    sns.countplot(x='Embarked', data=train_data, ax=ax[0,0])
    sns.countplot(x='Embarked', hue='Sex',data=train_data, ax=ax[0,1])
    sns.countplot(x='Embarked', hue='Survived', data=train_data, ax=ax[1, 0])
    sns.countplot(x='Embarked', hue='Pclass', data=train_data, ax=ax[1, 1])
    plt.show()

    # factorplot 으로 Embarked 와 Pclass 에 따른 Sex 생존율 보기
    # 여러 특성으로 데이터를 볼 때 factorplot 사용
    sns.factorplot('Pclass', 'Survived',hue='Sex',col='Embarked', data=train_data)
    plt.show() # 데이터를 보면 상관 관계가 보이지는 않는다.
    # S 에서 탄 사람들이 3등석 사람들이 많고
    # C 에서 탄 사람들의 3등석 비율은 적어서 비교적 C가 더 생존한 것처럼 보일 뿐 별 특별한 의미는 없다.

    # S 가 제일 많이 탑승한 곳이므로, 데이터에 영향이 덜 가도록 Nan 값을 S 로 대체한다.
    train_data['Embarked'].fillna('S', inplace=True)
    test_data['Embarked'].fillna('S', inplace=True)
    print(train_data['Embarked'].isnull().any())

#embarked_survived(train_data)



# SibSip (형재, 배우자 수)
# Discrete Feature

# 1) 생존과 SibSip 와의 관계를 본다.

def SibSp_Survive(train_data):
    print(pd.crosstab([train_data.SibSp], train_data.Survived))

    sns.barplot(x='SibSp', y='Survived', data=train_data)
    plt.show()
    sns.factorplot(x='SibSp', y='Survived', data=train_data)
    plt.show()

    print(pd.crosstab([train_data.SibSp], train_data.Pclass))
    # 가족수가 한 두명이면 생존 확률이 높지만, 가족수가 많을 경우 생존 확률이 매우 적다.
    # 가족수가 많을 때 3등석에 많이 가 있는 경우 때문이라고 본다.


#SibSp_Survive(train_data)



# Parch (부모, 자녀)
# Discrete data
def Parch_Survived(train_data):
    print(pd.crosstab(train_data.Parch, train_data.Survived))
    print(pd.crosstab(train_data.Parch, train_data.Pclass))

    sns.barplot(x='Parch', y='Survived', data=train_data)
    plt.show()
    sns.barplot(x='Parch', y='Survived', hue='Pclass', data=train_data)
    plt.show()
    sns.factorplot(x='Parch', y='Survived', data=train_data)
    plt.show() # 부모나 자녀의 수가 적절하면 생존률이 늘어나지만 너무 많으면 줄어든다.

#Parch_Survived(train_data)



# Fare

# Continous Feature (age 와 비슷)

# 최대, 평균, 최소 값을 본다.
#print('Highest Fare was:',train_data['Fare'].max())
#print('Lowest Fare was:',train_data['Fare'].min())
#print('Average Fare was:',train_data['Fare'].mean())

def Fare_Pclass(train_data):
    # 가격의 등석에 대한 분산을 본다.
    fig ,ax = plt.subplots(1,3,figsize=(10,5))
    sns.distplot(train_data[train_data['Pclass'] == 1].Fare , ax=ax[0])
    sns.distplot(train_data[train_data['Pclass'] == 2].Fare, ax=ax[1])
    sns.distplot(train_data[train_data['Pclass'] == 3].Fare, ax=ax[2])
    plt.show()
    # Pclass1 에 대한 분산이 매우 크다. 이 값들이 continous 하기 때문에 범위로 discrete 하게 만들 수 있다.

#Fare_Pclass(train_data)


'''
분석 결과;

sex: 성별이 여자일수록 생존 확률이 늘어난다.

Pclass: 1등석일수록 생존확률이 늘어난다. 특히 여성의 경우 1등석이면서 여성인 사람들이 많이 생존했다.

Age: 5~10세의 아이들이 많이 생존했고, 15~35살의 사람들이 생존하지 못 했다.

Embarked: C가 생존률이 높아 보이지만, 단지 3등석 비율이 적어서 생존률이 높아보이는 것이었다.

Parch + SibSp: 가족 수가 1~3이면 생존 확률이 높지만, 너무 많으면 생존 확률이 떨어진다
'''


# Feature 들 간의 상관관계 시각화
# redundant(동일, 중복)한 feature를 없애기 위해 상관관계를 조사한다.
# redundant 한 데이터를 찾아서 삭제해야한다.
# ex) 100개의 feature가 있는데 동일한 feature가 10개가 있다.
# 10개의 데이터가 그렇게 중요하진 않다.
# 그런데, tree 알고리즘을 사용하는데 10개의 feature가 tree를 가르는 feature 로 쓰이게 된다.
# 이런 경우 아예 10개의 feature를 없애야한다.
# 이런 경우를 해결하기 위해 상관관계를 조사한다.
def data_heatmap(train_data):
    sns.heatmap(data=train_data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2)
    fig = plt.gcf() # 현재 figure를 가져온다.
    fig.set_size_inches(10,8)
    plt.show()
    # 위 heatmap을 보면 redundant한 데이터는 없다는 것을 알 수 있다.
    # 만약 타깃값과 특성 값의 상관관계도가 1에 가까울 경우 -> LinearRegression 모델에 아주 강력한 특성이 될 수 있다.



# Part2: Feature Engineering and Data Cleaning:
# Feature Engineering
# 필요 없는 특성들은 제거하고 특성들 사이의 관계를 이용해 더 유용한 특성들을 만들어 본다.
# ex) 이름 feature에서 성 가져오기
# 또한 모델에 맞게 data 들을 변형 시켜야한다.


# 1. age 특성 구간별 데이터로 변형
# 최소가 0 최대가 80 이고 5개의 구간으로 나누면 16 씩 증가시키면 된다.

train_data['Age_band'] = 0
test_data['Age_band'] = 0
def slice_age(train_data):
    train_data.loc[train_data['Age'] <= 16, 'Age_band'] = 0
    train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age_band'] = 1
    train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age_band'] = 2
    train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age_band'] = 3
    train_data.loc[train_data['Age'] > 64, 'Age_band'] = 4

    test_data.loc[test_data['Age'] <= 16, 'Age_band'] = 0
    test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age_band'] = 1
    test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age_band'] = 2
    test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age_band'] = 3
    test_data.loc[test_data['Age'] > 64, 'Age_band'] = 4

    # label 과의 관계 조사해보기
    #sns.factorplot('Age_band', 'Survived', col='Pclass', data=train_data)
    #plt.show()
slice_age(train_data)



# 2. 가족 크기와 혼자 여부

train_data['Family_size'] = 0
train_data['Alone'] = 0

test_data['Family_size'] = 0
test_data['Alone'] = 0

def Family_size_alone(train_data):
    train_data['Family_size'] = train_data['SibSp'] + train_data['Parch']
    train_data.loc[train_data.Family_size == 0, 'Alone'] = 1

    test_data['Family_size'] = test_data['SibSp'] + test_data['Parch']
    test_data.loc[test_data.Family_size == 0, 'Alone'] = 1
    '''
    sns.factorplot(x='Family_size', y='Survived', data=train_data)
    plt.title('Family_size vs Survived')
    plt.show()
    sns.factorplot(x='Alone', y='Survived', data=train_data)
    plt.title('Alone vs Survived')
    plt.show()
    sns.factorplot(x='Alone', y='Survived', hue='Sex', data=train_data, col='Pclass')
    plt.show() # 3등석 여성 중 혼자 사는 사람의 생존율이 더 높은 것을 볼 수 있다.
    '''

Family_size_alone(train_data)





# 3. Fare_range
# 요금이 연속적인 데이터이기 때문에 이 또한 구간별로 나누어주어야 한다.
# pandas.qcut 을 이용한다. -> bin 매개변수 만큼 동등한 비율로 값을 나눈다.

def Fare_range(train_data):
    train_data['Fare_Range'] = pd.qcut(train_data['Fare'], 4)
    test_data['Fare_Range'] = pd.qcut(train_data['Fare'], 4)
    print(train_data['Fare_Range'])
    print(train_data.groupby(['Fare_Range'])['Survived'].mean())
    train_data['Fare_cat'] = 0
    test_data['Fare_cat'] = 0


    train_data.loc[train_data['Fare'] <= 7.91, 'Fare_cat'] = 0
    train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare_cat'] = 1
    train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31),'Fare_cat'] = 2
    train_data.loc[(train_data['Fare'] > 31) & (train_data['Fare'] <= 512.329), 'Fare_cat'] = 3

    test_data.loc[test_data['Fare'] <= 7.91, 'Fare_cat'] = 0
    test_data.loc[(test_data['Fare'] > 7.91) & (test_data['Fare'] <= 14.454), 'Fare_cat'] = 1
    test_data.loc[(test_data['Fare'] > 14.454) & (test_data['Fare'] <= 31), 'Fare_cat'] = 2
    test_data.loc[(test_data['Fare'] > 31) & (test_data['Fare'] <= 512.329), 'Fare_cat'] = 3

    #sns.factorplot(x='Fare_cat', y='Survived', data=train_data, hue='Sex')
    #plt.show()

Fare_range(train_data) # 값이 비쌀수록 살아남을 확률이 높다.
# 성별 데이터와 함께 사용하면 중요한 특성이 될 수 있다.





# 4. 문장을 수로 바꾸기

def convert_string_num(train_data):
    train_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    train_data['Embarked'].replace(['S', 'C', 'Q'], [0,1,2], inplace=True)
    train_data['initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4], inplace=True)

    test_data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    test_data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    test_data['initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)


convert_string_num(train_data)



# 5. 필요 없는 데이터 없애기

'''
Name -> 범주형 데이터로 변환시키지 못 해서 버린다.

Age -> Age_band를 만들었으므로 버린다.

Ticket -> 범주형 데이터로 변환시키지 못 해서 버린다.

Fare -> Fare_cat 을 만들었으므로 버린다.

Cabin -> 너무 많은 NaN 값이 있어 버린다.

Fare_Range -> fare_cat 이 있어 버린다.

PassengerId -> 범주화 될 수 없어 버린다.
'''
test_passenger_id = pd.Series(test_raw_data['PassengerId'])
def drop_data(train_data,test_data):
    train_data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)
    test_data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range', 'PassengerId'], axis=1, inplace=True)
    '''
    sns.heatmap(train_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':15})
    fig = plt.gcf()
    fig.set_size_inches(18,15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    '''

drop_data(train_data,test_data) # 큰 상관관계수를 갖는 특성들을 볼 수 있다.
# feature engineering 을 통해 더 상광관계수가 큰 특성을 찾았다.
print(train_data.shape)




# Part3: Predictive Modeling
# 분류 알고리즘을 사용하여 예측한다.

'''
사용해볼 알고리즘

1. logistic regression

2. SVM (linear, rbf)

3. random forest

4. kNN

5. naive bayes

6. decision tree

'''

# train data, test data 나누기

from sklearn.model_selection import train_test_split

train, test = train_test_split(train_data, test_size=0.3, random_state=0, stratify=train_data['Survived']) # stratify로 label 지정
train_x = train[train.columns[1:]]
train_y = train[train.columns[:1]]
test_x = test[test.columns[1:]]
test_y = test[test.columns[:1]]

X = train_data[train.columns[1:]] # cross validation 을 위해 사용한다.
Y = train_data['Survived']


# 모델 훈련
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def models(train_x, train_y, test_x,test_y):
    # SVC with rbf kernel
    model = SVC(kernel='rbf', C=1, gamma=0.1) # C 는 슬랙변수의 비중 조절, gamma는 종 모양의 크기 결정
    model.fit(train_x, train_y)
    prediction1 = model.predict(test_x)
    print('Accuracy for rbf SVM is ', accuracy_score(prediction1, test_y)) # Accuracy for rbf SVM is  0.835820895522388

    # Linear SVC
    model = SVC(kernel='linear', C=0.1, gamma=0.1)
    model.fit(train_x, train_y)
    prediction2 = model.predict(test_x)
    print('Accuracy for linear SVM is', accuracy_score(prediction2, test_y)) # Accuracy for linear SVM is 0.8171641791044776

    # Logistic Regression
    model = LogisticRegression()
    model.fit(train_x, train_y)
    prediction3 = model.predict(test_x)
    print('The accuracy of the Logistic Regression is', accuracy_score(prediction3, test_y)) # The accuracy of the Logistic Regression is 0.8208955223880597


    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    prediction4 = model.predict(test_x)
    print('The accuracy of the Decision Tree is', accuracy_score(prediction4, test_y)) # The accuracy of the Decision Tree is 0.8097014925373134

    # KNN
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    prediction5 = model.predict(test_x)
    print('The accuracy of the KNN is', accuracy_score(prediction5, test_y))
    # KNN 모델의 n_neighbours 를 변화 시키면 accuracy에 변화가 생긴다.

    '''
    # n_neighbours 에 따른 accuracy 변화를 관찰할 수 있다.
    a = []
    x = []
    for i in range(1,11):
        x.append(i)
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        a.append(accuracy_score(prediction, test_y))

    plt.plot(x,a)
    plt.xticks(x)
    plt.show()
    '''

    # Guassian Naive Bayes
    model = GaussianNB()
    model.fit(train_x, train_y)
    prediction6 = model.predict(test_x)
    print('The accuracy of the NaiveBayes is', accuracy_score(prediction6, test_y)) # The accuracy of the NaiveBayes is 0.8134328358208955


    # Random forest
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    prediction7 = model.predict(test_x)
    print('The accuracy of the Random Forests is', accuracy_score(prediction7, test_y)) # The accuracy of the Random Forests is 0.8134328358208955


models(train_x,train_y, test_x, test_y)



# cross validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

xyz = []
accuracy = []
std = []

kfold = KFold(n_splits=10, random_state=42)
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
modelss=[SVC(kernel='linear'),SVC(kernel='rbf'),LogisticRegression(),
        KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),
        GaussianNB(),RandomForestClassifier(n_estimators=100)]

for i in modelss:
    model=i
    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)

new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz,'Std':std}, index=classifiers)
print(new_models_dataframe2)

'''
plt.subplots(figsize=(12,6))
box = pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()
plt.show()
'''
'''
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
'''


# accuracy는 불균형때문에 오해의 소지가 있을 수 있다.
# 오차 행렬 (confusion matrix)를 통해 모델이 어떤 것을 잘못 예측했는지 알 수 있다.\
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix():
    f,ax=plt.subplots(3,3,figsize=(12,10))
    y_pred = cross_val_predict(SVC(kernel='rbf'),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
    ax[0,0].set_title('Matrix for rbf-SVM')
    y_pred = cross_val_predict(SVC(kernel='linear'),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
    ax[0,1].set_title('Matrix for Linear-SVM')
    y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
    ax[0,2].set_title('Matrix for KNN')
    y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
    ax[1,0].set_title('Matrix for Random-Forests')
    y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
    ax[1,1].set_title('Matrix for Logistic Regression')
    y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
    ax[1,2].set_title('Matrix for Decision Tree')
    y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
    sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
    ax[2,0].set_title('Matrix for Naive Bayes')
    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    plt.show()

# Naive Bayes  정밀도:0.72376, 재현율 0.76608

# rbf-SVM  정밀도: 0.80984, 재현율: 0.7222

# Naive bayes 는 살은 사람을 잘 분별하고
# rbf-svm 은 죽은 사람을 잘 분별한다.



# hyper parameter tuning

from sklearn.model_selection import GridSearchCV

# SVM
def Gridsearch_SVM():
    C = [0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
    gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    kernel = ['rbf', 'linear']
    hyper = {'kernel':kernel, 'C':C, 'gamma':gamma}
    gd = GridSearchCV(estimator=SVC(), param_grid=hyper, verbose=True)
    gd.fit(X,Y)
    print(gd.best_score_)
    print(gd.best_estimator_)
    print(gd.best_params_) # {'C': 0.6, 'gamma': 0.1, 'kernel': 'rbf'}


# RandomForest

def Gridsearch_RandomForest():
    n_estimators = range(100,1000,100)
    hyper = {'n_estimators': n_estimators}
    gd = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=hyper, verbose=True)
    gd.fit(X,Y)
    print(gd.best_score_) # 0.8148264390182665
    print(gd.best_estimator_)
    print(gd.best_params_) # {'n_estimators': 600}

#Gridsearch_RandomForest()






# Ensembling
# 여러 모델들을 모아 더 강력한 모델을 만든다.


# 1. Voting classifier

# 머신러닝 모델들의 예측들을 모으는 가장 간단한 방법이다.
# 서브모델들은 모두 다른 모델들이며 모든 서브모델들의 예측을 기반으로 평균 예측을 낸다.
'''
from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                                 ('RBF', SVC(probability=True , kernel='rbf', C=0.5, gamma=0.1)),
                                                 ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                                ('LR', LogisticRegression(C=0.05)),
                                                ('DT', DecisionTreeClassifier(random_state=0)),
                                                ('NB', GaussianNB()),
                                                ('svm', SVC(kernel='linear', probability=True))], voting='soft').fit(train_x, train_y)

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_x, test_y))
# The accuracy for ensembled model is: 0.8246268656716418

print('The cross validated score is',cross_val_score(ensemble_lin_rbf, X,Y,cv=10, scoring='accuracy').mean())
# The cross validated score is 0.8249188514357053
'''


# 2. Bagging
# 같은 모델을 여러 개 이용하고 같은 훈련 샘플을 여러 번 중복을 허용해서 샘플링하여 여러 개의 예측기를 훈련 시킨다.
# 모든 예측기가 훈련을 마치면 앙상블은 모든 예측기의 예측을 모아서 새로운 훈련 샘플에 대한 예측을 만든다.

# KNN Bagging

from sklearn.ensemble import BaggingClassifier
'''
model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3), random_state=42, n_estimators=700, oob_score=True)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print('The accuracy for bagged KNN is:',accuracy_score(prediction,test_y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())
print(model.oob_score_) # 별도의 검증세트 없이 oob score를 이용하여 평가할 수 있다.
'''


# Bagged Decision Tree
'''
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42, n_estimators=100)
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print('The accuracy for bagged Decision Tree is:',accuracy_score(prediction, test_y))
# The accuracy for bagged Decision Tree is: 0.835820895522388
result = cross_val_score(model, X, Y, scoring='accuracy', cv=10)
print('The cross validated score for bagged Decision Tree is:',result.mean())
# The cross validated score for bagged Decision Tree is: 0.8148564294631709
'''




# 3. Boosting
# 앞의 모델을 보완해나가면서 일련의 예측기를 학습시키는 것이다.
# 처음에는 모든 데이터를 학습하고
# 다음에는 잘못 예측된 데이터의 가중치를 높혀 잘못 예측된 데이터를 학습하도록 해준다.


# 가장 많이 사용되는 boosting은 Adaboost이다.
# base estimator는 decision tree로 이루어져있다.
# 그러나 base estimator 는 우리가 정할 수 있다.
'''
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=200, random_state=42, learning_rate=0.1)
result = cross_val_score(ada, X,Y, cv=10, scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())
'''

# 인기가 높은 또 하나의 부스팅 알고리즘은 그래디언트 부스팅이다.
# base estimator는 decision tree이다.
'''
from sklearn.ensemble import GradientBoostingClassifier

grad = GradientBoostingClassifier(n_estimators=500, random_state=42, learning_rate=0.1)
result = cross_val_score(grad, X,Y,scoring='accuracy', cv=10)
print('The cross validated score for Gradient Boosting is:', result.mean())
'''


# XGBoost
# extreme boosting 이다.
# GBM 을 기반으로 하고, GBM 의 단점인 느린 수행 시간과 과적합 규제 부재 등의 문제를 해결해서 각광받고 있다.
'''
import xgboost as xg

xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
result = cross_val_score(xgboost, X, Y, cv=10, scoring='accuracy')
print('The cross validated score for XGBoost is:', result.mean())
'''

# hyper-parameter tuning
'''
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

hyper = {'n_estimators':n_estimators, 'learning_rate':learn_rate}
gd = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True)
gd.fit(X,Y)
print(gd.best_score_) # 0.8271483271608814
print(gd.best_estimator_)
'''


#AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.05,
#                   n_estimators=300, random_state=None)

#print(gd.best_params_) #  {'learning_rate': 0.05, 'n_estimators': 300}




# Confusion Matrix for the Best Model
'''
ada = AdaBoostClassifier(n_estimators=300, random_state=42, learning_rate=0.05)
result = cross_val_predict(ada, X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, result), cmap='winter', annot=True, fmt='2.0f')
plt.show()
'''


# Feature Impotance
'''
f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()
'''

# initial이 sex 와 관련이 있어서 feature importance 가 높게 나온다.
# 대신 sex의 feature importance 가 적게 나온다.
# Pclass, Fare_cat, 가족 수의 feature importance 가 크게 나오는 것을 볼 수 있다.





# 제출

from sklearn.ensemble import AdaBoostClassifier

def make_result_csv(test_raw_data, result):
    result_dataframe = pd.DataFrame(
        {
            "PassengerId": test_passenger_id.to_numpy(),
            "Survived": result
        }
    )
    result_dataframe = result_dataframe.set_index("PassengerId")
    result_dataframe.to_csv("submission.csv")

model = AdaBoostClassifier(n_estimators=300, random_state=42, learning_rate=0.05)
model.fit(X, Y)
result = model.predict(test_data)
make_result_csv(test_raw_data, result)


'''
def make_result_csv(result):
    result_dataframe = pd.DataFrame(
        {
            "PassengerId": test_passenger_id.to_numpy(),
            "Survived": result
        }
    )
    result_dataframe = result_dataframe.set_index("PassengerId")
    result_dataframe.to_csv("submission.csv")

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100, input_dim=10,kernel_initializer='he_normal', activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=50, kernel_initializer='he_normal', activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=25, kernel_initializer='he_normal', activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=10, kernel_initializer='he_normal', activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1, kernel_initializer='he_normal', activation='sigmoid')
])

model.compile(optimizer=tf.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999),loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=32)
result = model.predict(test_data)

from sklearn.preprocessing import Binarizer
import numpy as np
binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(result)
test_predict_result=test_predict_result.astype(np.int32)

model.evaluate(test_x, test_y)

make_result_csv(test_predict_result.ravel())
'''

```
