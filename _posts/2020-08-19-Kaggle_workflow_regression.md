---
layout: post
title: Kaggle workflow Regression
description: " Kaggle workflow Regression"
modified: 2020-08-11
tags: [Kaggle]
categories: [Kaggle]
---

## 이상치 탐지를 비지도 학습으로 
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 가져오기
def get_data():
    train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

    #print(train_df.head(5))
    #print(test_df.head(5))
    return train_df, test_df

train_raw_data, test_raw_data = get_data()


# Label 확인

def glance_label(train_raw_data):
    print(train_raw_data['SalePrice'])
    sns.distplot(train_raw_data['SalePrice'])
    plt.show()

#glance_label(train_raw_data)

# ****** 변수가 많을 때 *****
# 1. 상관관계수를 조사한다
# 2. 상관관계수가 높은 데이터를 뽑아낸다.

# 고려할 변수가 많을 땐 상관관계를 검토하여 상관관계가 높은 변수를 검토해본다.
#print(train_raw_data.corr())
corr = train_raw_data.corr()
#print(corr.nlargest(50, 'SalePrice')['SalePrice'])

def correlation(train_raw_data):
    corr = train_raw_data.corr()
    indexes = corr.nlargest(40, 'SalePrice')['SalePrice'].index
    print(corr.nlargest(40, 'SalePrice')['SalePrice'])
    sns_plot = sns.heatmap(train_raw_data[indexes].corr(), annot=True, cmap='RdYlGn', linewidths=0.2,
                xticklabels=indexes, yticklabels=indexes)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    sns_plot.get_figure().savefig('Corr')

#correlation(train_raw_data)


#train_data, test_data = drop_data(train_raw_data, test_raw_data)


# 상관관계가 높은 데이터들을 시각화 해본다.

def pair_plot(train_raw_data):
    sns.pairplot(train_raw_data[corr.nlargest(7, 'SalePrice')['SalePrice'].index],size=1)
    plt.show()


#pair_plot(train_raw_data) # 선형적인 상관관계를 볼 수 있다.


def scatter_plots(train_raw_data):
    indexes = corr.nlargest(8, 'SalePrice')['SalePrice'].index
    plt.subplots(figsize=(15,10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for num, i in enumerate(indexes[1:]):
        plt.subplot(240 + num + 1)
        sns.scatterplot(x=i, y=indexes[0], data=train_raw_data)
    plt.show()

#scatter_plots(train_raw_data)




# 결측값 확인
# 보통 15% 이상의 결측값이 있으면 특성을 삭제해버린다.
def frauds(train_raw_data, test_raw_data):
    #print(train_raw_data.isnull().sum())
    #print("train data 결측값:")
    total = train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum().sort_values(ascending=False)
    percentage = (train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum()/len(train_raw_data)).sort_values(ascending=False)
    train_data_fruad = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
    #print(train_data_fruad)
    #print()
    #print("test data 결측값:")
    test_total = test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum().sort_values(ascending=False)
    test_percentage = (test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum()/len(test_raw_data)).sort_values(ascending=False)
    test_data_fruad = pd.concat([test_total, test_percentage], axis=1, keys=['Total', 'Percent'])
    #print(test_data_fruad)
    '''
    fig, ax = plt.subplots(figsize=(10,5))
    train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum().plot.bar()
    plt.show()
    test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum().plot.bar()
    plt.show()
    '''
    return train_data_fruad, test_data_fruad

train_data_fruad, test_data_fruad = frauds(train_raw_data, test_raw_data)

def find_object_fraud_train(train_data_fruad, object):
    object_indexes = []
    for i in train_data_fruad.index:
        if i.startswith(object):
            object_indexes.append(i)
    return object_indexes


def find_object_fraud_test(test_data_fruad, object):
    object_indexes = []
    for i in test_data_fruad.index:
        if i.startswith(object):
            object_indexes.append(i)
    return object_indexes


# 결측값 없애기
# 결측률이 15% 이상인 데이터는 모두 삭제한다.
def train_preprocess(train_raw_data, train_data_fraud):
    train_raw_data.drop(train_data_fruad[(train_data_fruad['Percent'] > 0.15)].index, axis=1, inplace=True)

    garage_indexes = find_object_fraud_train(train_data_fruad, 'Garage')
    train_raw_data.drop(garage_indexes, axis=1, inplace=True)

    Bsmt_indexes = find_object_fraud_test(train_data_fruad, 'Bsmt')
    train_raw_data.drop(Bsmt_indexes, axis=1, inplace=True)

    MasVnr_indexes = find_object_fraud_train(train_data_fruad, 'MasVnr')
    train_raw_data.drop(MasVnr_indexes, axis=1, inplace=True)

    # Electrical은 결측 데이터가 하나이므로 그냥 삭제
    train_raw_data.drop(test_raw_data.loc[train_raw_data['Electrical'].isnull()].index, axis=0, inplace=True)

    # Id 없애기
    train_raw_data.drop('Id', axis=1, inplace=True)

    train_data = train_raw_data

    return train_data

test_indexes = test_raw_data['Id']
#print(test_indexes.to_numpy())

def test_preprocess(test_raw_data, test_data_fraud):
    test_raw_data.drop(train_data_fruad[(train_data_fruad['Percent'] > 0.15)].index, axis=1, inplace=True)

    # GarageCars 로 데이터 대채 가능
    garage_indexes = find_object_fraud_test(train_data_fruad, 'Garage')
    test_raw_data.drop(garage_indexes, axis=1, inplace=True)

    # TotalBsmtSF 로 대채 가능
    Bsmt_indexes = find_object_fraud_test(train_data_fruad, 'Bsmt')
    test_raw_data.drop(Bsmt_indexes, axis=1, inplace=True)

    # MasVnr 의 상관관계도가 영향을 미미하게 미치고 다른 특성으로 대체 가능
    MasVnr_indexes = find_object_fraud_test(train_data_fruad,'MasVnr')
    test_raw_data.drop(MasVnr_indexes, axis=1, inplace=True)

    # 'Id' 없애기
    test_raw_data.drop('Id', axis=1, inplace=True)
    test_data = test_raw_data

    # 나머지 데이터 평균으로 채우기

    for col in test_data[test_data.columns[test_data.isnull().sum() > 0]].columns:

        if type(test_data[col].iloc[0]) == np.str:
            test_data[col].fillna(test_data[col].value_counts().index[0], inplace=True)
        else:
            test_data[col].fillna(np.mean(test_data[col]), inplace=True)
    return test_data



train_data = train_preprocess(train_raw_data, train_data_fruad)
test_data = test_preprocess(test_raw_data, test_data_fruad)
frauds(train_raw_data, test_raw_data)
print("train_data 결측값 수:", train_raw_data.isnull().sum().max())



#----------------------------------------------------------------------------------------
from sklearn.cluster import DBSCAN


list_for_outlier = []
for col in train_data.columns:
    if type(train_data[col].iloc[0]) == np.str:
        continue
    else:
        list_for_outlier.append(col)
print(list_for_outlier)


dbscan = DBSCAN(eps=10000, min_samples=5)

dbscan.fit(train_data[list_for_outlier])
print(set(dbscan.labels_))


def drop_outlier(dbscan, X):
    outlier = (dbscan.labels_ == -1)
    outlier_columns = X[outlier]
    print(outlier_columns.index)
    train_data.drop(outlier_columns.index, axis=0,inplace=True)


'''
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    outlier = (dbscan.labels_ == -1)
    outlier_columns = X[outlier]
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(outlier_columns[:,0], outlier_columns[:,1], marker='x', c='r', s=20)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

def plot_after_drop(X):
    plt.scatter(X[:, 0], X[:, 1])


plt.figure(figsize=(9, 3.2))

x = pd.concat([train_data['TotalBsmtSF'],train_data['SalePrice']], axis=1)
x = x.to_numpy()


plot_dbscan(dbscan, x, size=100)
plt.savefig('dbscan/before')
plt.show()
'''

drop_outlier(dbscan=dbscan, X=train_data)


'''
x = pd.concat([train_data['TotalBsmtSF'],train_data['SalePrice']], axis=1)
x = x.to_numpy()
plt.figure(figsize=(9, 3.2))
plot_after_drop(x)
plt.savefig('dbscan/after')
plt.show()
'''


from scipy import stats
from scipy.stats import norm



#------------------------------------------------------------------------------------------------

def Check_Normality_label(train_data):
    '''
    sns.distplot(train_data['SalePrice'], fit=norm)
    plt.show()
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    sns.distplot(train_data['SalePrice'], label="Skewness: {:2f}".format(train_data["SalePrice"].skew(), ax=ax))
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['SalePrice'], plot=plt)
    plt.show()
    '''

    print("Skewness: %f" % train_data["SalePrice"].skew()) # skewness 측정
    print("Kurtosis: %f" % train_data["SalePrice"].kurt()) # peakedness 측정

    # SalePrice 가 정규분포를 따르지 않는다.
    # 이 데이터는 뾰족(peakedness)하고 positive skewness 를 가지고 있다. 또한 대각선을 따르고 있지 않다.
    # (Kurtosis 가 peakedness(높다) 하면 데이터가 heavy tails 를 가지고 있거나 outliers가 많다는 뜻이다. -> 반드시 잡아야한다.)
    # (Low kurtosis 를 가지고 있으면 light tails 를 가지고 있거나 outliers가 적다는 뜻이다. -> 데이터 세트를 조사하고 정리해야한다.)

    # 간단한 데이터 변환이 이 문제를 해결해준다.
    # 이것은 통계 서적에서 배울 수 있는 놀라운 것 중 하나이다.
    # positive skewness 이 있는 경우 log 변환이 일반적으로 잘 작동한다.

    train_data['SalePrice'] = np.log(train_data['SalePrice'])

    '''
    sns.distplot(train_data['SalePrice'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['SalePrice'], plot=plt)
    plt.show()
    '''
    # 잘 해결된 것을 볼 수 있다.

Check_Normality_label(train_data)


# 'GrLivArea' 에 대해서 살펴본다.

def Check_GrLivArea_Normality(train_data):
    '''
    sns.distplot(train_data['GrLivArea'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['GrLivArea'], plot=plt)
    plt.show()
    '''

    train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])
    '''
    sns.distplot(train_data['GrLivArea'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['GrLivArea'], plot=plt)
    plt.show()
    '''

Check_GrLivArea_Normality(train_data)

#scatterplot_with_SalePrice('GrLivArea',train_data)# 더이상 특정 모양을 띄지 않는다. 이게 정규화의 힘이다.
# 이는 동분산성 문제를 해결해 준다

# TotalBsmtSF에 대해 살펴본다.

def Check_TotalBsmtSF_Normality(train_data):
    '''
    sns.distplot(train_data['TotalBsmtSF'], fit=norm)
    plt.show()
    fig = plt.figure()
    res  = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
    plt.show()
    '''
    # 데이터가 skewness 를 보인다.
    # 데이터에서 꽤 많은 0 데이터(basement 가 없는 데이터)를 관찰할 수 있다.
    # 이 데이터는 중간에 0 값이 있어서 log 변환을 하지 못한다.

    # log 변환 시킬 수 있는 방법
    # basement 의 유무를 판단할 수 있는 colunm 을 만들고 0이 아닌 데이터에 관해서만 log 변환을 취한다.
    # 이 방법이 basement 가 없는 효과를 잃지 않고 데이터를 변환 시킬 수 있는 방법이다.
    train_data['HasBsmt'] = 0
    train_data.loc[train_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    train_data.loc[train_data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train_data[train_data['HasBsmt'] == 1]['TotalBsmtSF'])

    test_data['HasBsmt'] = 0
    test_data.loc[test_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    test_data.loc[test_data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(
        test_data[test_data['HasBsmt'] == 1]['TotalBsmtSF'])

    # 0 보다 큰 값들만 그래프로 그린다. 왜냐하면, 0보다 큰 값들만 log 변환시켰기 때문에 0보다 큰 값들에 대해서만 본다.
    '''
    sns.distplot(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
    plt.show()
    '''

Check_TotalBsmtSF_Normality(train_data)

#sns.scatterplot(x=train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], y=train_data[train_data['TotalBsmtSF']>0]['SalePrice'], data=train_data)

# 더이상 특정 모양을 띄지 않는다. 이게 정규화의 힘이다.
# 이는 동분산성 문제를 해결해 준다. (즉, SalePrice 는 TotalBsmtSF 범위에서 동일한 수준의 분산을 나타낸다.)

#plt.show()



# train data Categorical data 수치형 변환


train_data.to_csv('train_data_to_see')
test_data.to_csv('test_data_to_see')




def get_string_data_colunms(train_data):
    string_colunms = []
    for data in train_data.columns:
        if type(train_data[data].iloc[0]) == np.str:
            string_colunms.append(data)
    return string_colunms

def drop_string_same_datas(train_data):
    idx = []
    for colunms in get_string_data_colunms(train_data):
        if len(train_data[colunms].value_counts()) == 1:
            train_data.drop(colunms, axis=1, inplace=False)


drop_string_same_datas(train_data)
ordinal_encoders_colunms = get_string_data_colunms(train_data)


from sklearn.preprocessing import OrdinalEncoder

def string_to_cat(train_data, colunms):
    ordinal_encoder = OrdinalEncoder()
    datas = train_data[[colunms]]
    Transposed = ordinal_encoder.fit_transform(datas)
    train_data[colunms] = Transposed.astype(np.int)


for colunms in ordinal_encoders_colunms:
    string_to_cat(train_data, colunms)


# test_data Categorical data 수치형 변환

drop_string_same_datas(test_data)
ordinal_encoders_colunms = get_string_data_colunms(test_data)
test_data.drop('HasBsmt', axis=1, inplace=True)
#(ordinal_encoders_colunms)

for colunms in ordinal_encoders_colunms:
    string_to_cat(test_data, colunms)


#print(len(get_string_data_colunms(train_data)))
#print(len(get_string_data_colunms(test_data)))


def column_test(train_data, test_data):
    for col in train_data:
        if col not in test_data:
            print(col)

column_test(train_data, test_data)


# model prediction

# LinearRegression, SVR, RandomForest, Xgboost, LightGBM

# split label, train data

def split_label(train_data):
    label = train_data[['SalePrice']]
    train = train_data.iloc[:,:-2]
    return train, label

x_train, y_train = split_label(train_data)



# model prediction

# LinearRegression, SVR, RandomForest, Xgboost, LightGBM

# split label, train data

def split_label(train_data):
    label = train_data[['SalePrice']]
    train = train_data.iloc[:,:-2]
    return train, label

x_train, y_train = split_label(train_data)

column_test(x_train, test_data)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

def Models(x_train, y_train):

    model_list = [(LinearRegression(n_jobs=-1),"LinearRegression") ,
                  (SVR(kernel='rbf',gamma=0.1, C=1,epsilon=0.1),"SVR"),
                  (RandomForestRegressor(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1),"RandomForestRegressor"),
                  (XGBRegressor(objective='reg:squarederror',n_estimators=1000, learning_rate=0.1,n_jobs=-1),"XGBRegressor"),
                  (AdaBoostRegressor(base_estimator=Ridge(alpha=2), n_estimators=1000, learning_rate=0.5), "AdaBoostRegressor"),
                  (GradientBoostingRegressor(max_depth=2, n_estimators=1100,),"GBM"),
                  (LGBMRegressor(boosting_type='gbdt',n_estimators=200, n_jobs=-1),"LGBM"),
                  (Ridge(alpha=1), "Ridge")
                  ]

    for model, name in model_list:
        train_model = model
        train_model.fit(x_train, y_train)
        rmse_score = cross_val_score(estimator=model, X=x_train, y=y_train, scoring="neg_mean_squared_error", cv=10)
        print(name,"RMSE value")
        print(np.sqrt(-rmse_score).mean())
        print()

#Models(x_train, y_train.values.flatten())


# 하이퍼 파라미터 튜닝

'''
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'max_depth': [2,3,4], 'n_estimators':[1000,1100,1200,1300],'learning_rate':[0.1,0.2,0.3]}
]

gbm = GradientBoostingRegressor(max_depth=2, n_estimators=1100)

grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=10, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(x_train, y_train.values.ravel())
print(np.sqrt(-grid_search.best_score_).mean())
print(grid_search.best_estimator_)
print(grid_search.best_params_)
'''

'''
0.11589065003265452
GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=2,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1100,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 1100}

'''


# 모델 선정
'''
model1 = XGBRegressor(objective='reg:squarederror',n_estimators=900, learning_rate=0.1,n_jobs=-1)
model1.fit(x_train, y_train)
y_predict = model1.predict(test_data)

result_df = pd.DataFrame({
    'Id': test_indexes.to_numpy(),
    'SalePrice': np.exp(y_predict)
})
result_df = result_df.set_index('Id')
result_df.to_csv("submission.csv")

print(result_df)
'''

import math

model1 = GradientBoostingRegressor(max_depth=2, n_estimators=1100,learning_rate=0.1)
model1.fit(x_train, y_train.values.ravel())
y_predict = model1.predict(test_data)

result_df = pd.DataFrame({
    'Id': test_indexes.to_numpy(),
    'SalePrice': np.exp(y_predict)
})
result_df = result_df.set_index('Id')
result_df.to_csv("submission.csv")

print(result_df)


# 신경망

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

#print(x_train.shape)
#print(x_test.shape)



import tensorflow as tf
import os
import datetime

# 너무 복잡도가 크면 안 됨...
# 오히려 신경망이 얕은 경우의 rmse 가 작음
'''
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq= 1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=600, input_dim=len(x_train.columns), activation='elu',
                                kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=600, input_dim=len(x_train.columns), activation='elu',
                                kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=500, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=500, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=500, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=250, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=500, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=250, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=250, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=250, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=100, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=100, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=50, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=25, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=1))

s = 500 * x_train.shape[0] // 32
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics='mse')


model.fit(x_train, y_train,batch_size=32, epochs=500, validation_split=0.2, shuffle=True)
evaluation = model.evaluate(x_test, y_test)
print(np.sqrt(evaluation))
'''

# 얕은 신경망
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=25, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=25, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=3, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.003)))
model.add(tf.keras.layers.Dense(units=1))

s = 2000 * x_train.shape[0] // 32
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics='mse')


model.fit(x_train, y_train, batch_size=32, epochs=4000, validation_split=0.2, shuffle=True)
evaluation = model.evaluate(x_test, y_test)
print(np.sqrt(evaluation))


# 자기 정규화는 잘 안 됨
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=50, input_dim=len(x_train.columns), activation='selu',
                                kernel_initializer='lecun_normal'))
model.add(tf.keras.layers.Dense(units=25, activation='selu', kernel_initializer='lecun_normal'))
model.add(tf.keras.layers.Dense(units=5, activation='selu', kernel_initializer='lecun_normal'))
model.add(tf.keras.layers.Dense(units=1))

s = 500 * x_train.shape[0] // 32
learning_rate1 = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate1, rho=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics='mse')

model.fit(x_train, y_train,batch_size=32, epochs=1000, validation_split=0.2,shuffle=True)
evaluation = model.evaluate(x_test, y_test)
print(np.sqrt(evaluation))
'''
```

<br>
<br>

## IQR 을 이용하여 이상치 탐지를 

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 가져오기
def get_data():
    train_df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test_df = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

    #print(train_df.head(5))
    #print(test_df.head(5))
    return train_df, test_df

train_raw_data, test_raw_data = get_data()


# Label 확인

def glance_label(train_raw_data):
    print(train_raw_data['SalePrice'])
    sns.distplot(train_raw_data['SalePrice'])
    plt.show()

#glance_label(train_raw_data)

# ****** 변수가 많을 때 *****
# 1. 상관관계수를 조사한다
# 2. 상관관계수가 높은 데이터를 뽑아낸다.

# 고려할 변수가 많을 땐 상관관계를 검토하여 상관관계가 높은 변수를 검토해본다.
#print(train_raw_data.corr())
corr = train_raw_data.corr()
#print(corr.nlargest(50, 'SalePrice')['SalePrice'])

def correlation(train_raw_data):
    corr = train_raw_data.corr()
    indexes = corr.nlargest(40, 'SalePrice')['SalePrice'].index
    print(corr.nlargest(40, 'SalePrice')['SalePrice'])
    sns_plot = sns.heatmap(train_raw_data[indexes].corr(), annot=True, cmap='RdYlGn', linewidths=0.2,
                xticklabels=indexes, yticklabels=indexes)
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    sns_plot.get_figure().savefig('Corr')

#correlation(train_raw_data)


#train_data, test_data = drop_data(train_raw_data, test_raw_data)


# 상관관계가 높은 데이터들을 시각화 해본다.

def pair_plot(train_raw_data):
    sns.pairplot(train_raw_data[corr.nlargest(7, 'SalePrice')['SalePrice'].index],size=1)
    plt.show()


#pair_plot(train_raw_data) # 선형적인 상관관계를 볼 수 있다.


def scatter_plots(train_raw_data):
    indexes = corr.nlargest(8, 'SalePrice')['SalePrice'].index
    plt.subplots(figsize=(15,10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for num, i in enumerate(indexes[1:]):
        plt.subplot(240 + num + 1)
        sns.scatterplot(x=i, y=indexes[0], data=train_raw_data)
    plt.show()

#scatter_plots(train_raw_data)




# 결측값 확인
# 보통 15% 이상의 결측값이 있으면 특성을 삭제해버린다.
def frauds(train_raw_data, test_raw_data):
    #print(train_raw_data.isnull().sum())
    #print("train data 결측값:")
    total = train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum().sort_values(ascending=False)
    percentage = (train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum()/len(train_raw_data)).sort_values(ascending=False)
    train_data_fruad = pd.concat([total, percentage], axis=1, keys=['Total', 'Percent'])
    #print(train_data_fruad)
    #print()
    #print("test data 결측값:")
    test_total = test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum().sort_values(ascending=False)
    test_percentage = (test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum()/len(test_raw_data)).sort_values(ascending=False)
    test_data_fruad = pd.concat([test_total, test_percentage], axis=1, keys=['Total', 'Percent'])
    #print(test_data_fruad)
    '''
    fig, ax = plt.subplots(figsize=(10,5))
    train_raw_data[train_raw_data.columns[train_raw_data.isnull().sum() > 0]].isnull().sum().plot.bar()
    plt.show()
    test_raw_data[test_raw_data.columns[test_raw_data.isnull().sum() > 0]].isnull().sum().plot.bar()
    plt.show()
    '''
    return train_data_fruad, test_data_fruad

train_data_fruad, test_data_fruad = frauds(train_raw_data, test_raw_data)

def find_object_fraud_train(train_data_fruad, object):
    object_indexes = []
    for i in train_data_fruad.index:
        if i.startswith(object):
            object_indexes.append(i)
    return object_indexes


def find_object_fraud_test(test_data_fruad, object):
    object_indexes = []
    for i in test_data_fruad.index:
        if i.startswith(object):
            object_indexes.append(i)
    return object_indexes


# 결측값 없애기
# 결측률이 15% 이상인 데이터는 모두 삭제한다.
def train_preprocess(train_raw_data, train_data_fraud):
    train_raw_data.drop(train_data_fruad[(train_data_fruad['Percent'] > 0.15)].index, axis=1, inplace=True)

    garage_indexes = find_object_fraud_train(train_data_fruad, 'Garage')
    train_raw_data.drop(garage_indexes, axis=1, inplace=True)

    Bsmt_indexes = find_object_fraud_test(train_data_fruad, 'Bsmt')
    train_raw_data.drop(Bsmt_indexes, axis=1, inplace=True)

    MasVnr_indexes = find_object_fraud_train(train_data_fruad, 'MasVnr')
    train_raw_data.drop(MasVnr_indexes, axis=1, inplace=True)

    # Electrical은 결측 데이터가 하나이므로 그냥 삭제
    train_raw_data.drop(test_raw_data.loc[train_raw_data['Electrical'].isnull()].index, axis=0, inplace=True)

    # Id 없애기
    train_raw_data.drop('Id', axis=1, inplace=True)

    train_data = train_raw_data

    return train_data

test_indexes = test_raw_data['Id']

def test_preprocess(test_raw_data, test_data_fraud):
    test_raw_data.drop(train_data_fruad[(train_data_fruad['Percent'] > 0.15)].index, axis=1, inplace=True)

    # GarageCars 로 데이터 대채 가능
    garage_indexes = find_object_fraud_test(train_data_fruad, 'Garage')
    test_raw_data.drop(garage_indexes, axis=1, inplace=True)

    # TotalBsmtSF 로 대채 가능
    Bsmt_indexes = find_object_fraud_test(train_data_fruad, 'Bsmt')
    test_raw_data.drop(Bsmt_indexes, axis=1, inplace=True)

    # MasVnr 의 상관관계도가 영향을 미미하게 미치고 다른 특성으로 대체 가능
    MasVnr_indexes = find_object_fraud_test(train_data_fruad,'MasVnr')
    test_raw_data.drop(MasVnr_indexes, axis=1, inplace=True)

    # 'Id' 없애기
    test_raw_data.drop('Id', axis=1, inplace=True)
    test_data = test_raw_data

    # 나머지 데이터 평균으로 채우기

    for col in test_data[test_data.columns[test_data.isnull().sum() > 0]].columns:

        if type(test_data[col].iloc[0]) == np.str:
            test_data[col].fillna(test_data[col].value_counts().index[0], inplace=True)
        else:
            test_data[col].fillna(np.mean(test_data[col]), inplace=True)
    return test_data



train_data = train_preprocess(train_raw_data, train_data_fruad)
test_data = test_preprocess(test_raw_data, test_data_fruad)
frauds(train_raw_data, test_raw_data)
#print("train_data 결측값 수:", train_raw_data.isnull().sum().max())




# 이상치 검출
# 이상치 검출을 하는 이유: 이상치가 우리 모델에 영향을 줄 수 있고 좋은 정보에 영향을 줄 수 있어서

# 이상치 탐지를 위해 데이터를 평준화한다. 여기서 평준화는 데이터가 평균이 0이고 표준편차가 1이게 만드는 것이다.

# 1. 수치적 이상치 검출 IQR (Inter Quanatile Range)
# -> 사분위 값의 편차를 이용한다. -> 이는 boxplot으로 확인 가능하다.

# 그래프 확인
'''
sns.scatterplot(x='GarageArea', y='SalePrice', data=train_data)
plt.show()
'''

# IQR 계산
#----------------------------------------
import numpy as np
import collections


def IQR_calc(train_data,n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(train_data[col], 25) # 데이터의 25% 지점
        Q3 = np.percentile(train_data[col], 75) # 데이터의 75% 지점
        IQR = Q3 - Q1 #(Q1 ~ Q3 까지의 범위를 IQR 이라고 한다.)

        outlier_step = IQR * 1.5
        # IQR 에 1.5를 곱해서 생성된 범위를 이용해 최댓값과 최솟값을 결정한다.

        outlier_list_col = train_data[(train_data[col] < Q1 - outlier_step) | (train_data[col] > Q3 + outlier_step)].index
        # IQR 최댓값보다 크거나 최솟값보다 작은 값을 뽑아낸다.
        outlier_indices.extend(outlier_list_col)
    outlier_indices = collections.Counter(outlier_indices)
    print(outlier_indices)
    outlier_to_drop = list(k for k, v in outlier_indices.items() if v > n)
    return outlier_to_drop

list_for_outlier = []
for col in train_data.columns:
    if type(train_data[col].iloc[0]) == np.str:
        continue
    else:
        list_for_outlier.append(col)
print(train_data.shape)
outlier_to_drop = IQR_calc(train_data,2, list_for_outlier)
train_data.drop(outlier_to_drop, axis=0, inplace=True)
print(train_data.shape)



'''
plot2 = sns.scatterplot(x='GarageArea', y='SalePrice', data=train_data)
plt.xlim(-50,1500)
plt.ylim(0,700000)
plt.show()
'''


# Skewness (비대칭도) 확인
# -> distplot 을 그려보고 치우친 데이터가 있으면 그 데이터를 고쳐준다.

# 이제 SalePrice 가 다변량 기법을 적용할 수 있는 통계적 가정을 어떻게 따르는지 깊이 이해해보자.

# 1. Normality
# -> 데이터가 정규분포와 비슷하게 생겨야한다.
#   여러 통계 테스트가 이에 의존하기 때문에 이것이 중요하다.
#   이번에는 'SalePrice' 에 대한 단지 일변량 정규성(univariate normality)을 체크할 것이다.
#   일변량 정규성은 다변량 정규성을 보장하지 않음을 기억하자. 그러나 도움은 된다.
#   고려해야할 또 다른 세부 사항은 큰 표본(200개 이상의 관측치)에서 정규성이 문제가되지 않는다는 것이다.
#   그러나 우리가 normality 문제를 해결한다면 다른 많은 문제들을 피할 수 있다.(이분산적)
#   이러한 이유로 Normality 분석이 필요하다.


# 2. Homoscedasticity(동질성)
# -> 종속변수가 예측 변수 범위에서 동일한 수준의 분산을 보인다는 가정을 의미한다.
# 오차항이 독립 변수의 모든 값에서 동일하기를 원하기 때문에 동질성이 바람직하다.


# 3. Linearity (선형성)
# 선형성을 평가하는 가장 일반적인 방법은 산점도를 조사하고 선형 패턴을 검색하는 것이다.
# 패턴이 선형이 아닌 경우 데이터 변환을 탐색하는 것이 좋다.
# 그러나 우리의 데이터는 대부분의 산점도가 선형 관계를 갖는 것처럼 보이기 때문에 이 부분은 안 할 것이다.

# 4. 상관 오류 없음
# 상관오류: 한 오류가 다른 오류와 연관될 때 발생한다.
# ex) 하나의 긍정적인 오류가 부정적인 오류를 만든다면 이는 변수 사이에 관계가 있음을 의미한다.
# 이것은 시계열에서 자주 발생하며 일부 패턴은 시간과 관련이 있다.
# 여기서는 다루지 않는다.
# 그러나 무언가를 감지하면 그 효과를 설명할 수 있는 변수를 추가해야한다.
# 이것이 상관 오류에 대한 가장 일반적인 솔루션이다.




# Normality 확인하기
# 여기서 요점은 매우 간결한 방식으로 'SalePrice' 를 테스트하는 것이다.

# 1. 히스토그램 - 첨도(Kurtosis)와 왜도(skewness)
# (첨도: 첨도는 분포의 꼬리부분의 길이와 중앙부분의 뾰족함에 대한 정보를 제공하는 통계량이다)
# (왜도: 왜도는 분포의 비대칭도를 나타내는 통계량이다)

# 2. Norm Probability plot
# -> 데이터 분포는 정규 분포를 나타내는 대각선을 가깝게 따라야한다.
from scipy import stats
from scipy.stats import norm

def scatterplot_with_SalePrice(n, traindata):
    sns.scatterplot(x=n,y='SalePrice', data=train_data)
    plt.show() # 콘 모양의 scatter 그래프를 볼 수 있다.

#scatterplot_with_SalePrice('GrLivArea',train_data) # 데이터가 콘 모양의 분포를 하고있다.
#scatterplot_with_SalePrice('TotalBsmtSF',train_data) # 데이터가 약간 다이아몬드 모양을 하고 있다.
# -> log 변환 시켜주기 전

def Check_Normality_label(train_data):
    '''
    sns.distplot(train_data['SalePrice'], fit=norm)
    plt.show()
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    sns.distplot(train_data['SalePrice'], label="Skewness: {:2f}".format(train_data["SalePrice"].skew(), ax=ax))
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['SalePrice'], plot=plt)
    plt.show()
    '''

    print("Skewness: %f" % train_data["SalePrice"].skew()) # skewness 측정
    print("Kurtosis: %f" % train_data["SalePrice"].kurt()) # peakedness 측정

    # SalePrice 가 정규분포를 따르지 않는다.
    # 이 데이터는 뾰족(peakedness)하고 positive skewness 를 가지고 있다. 또한 대각선을 따르고 있지 않다.
    # (Kurtosis 가 peakedness(높다) 하면 데이터가 heavy tails 를 가지고 있거나 outliers가 많다는 뜻이다. -> 반드시 잡아야한다.)
    # (Low kurtosis 를 가지고 있으면 light tails 를 가지고 있거나 outliers가 적다는 뜻이다. -> 데이터 세트를 조사하고 정리해야한다.)

    # 간단한 데이터 변환이 이 문제를 해결해준다.
    # 이것은 통계 서적에서 배울 수 있는 놀라운 것 중 하나이다.
    # positive skewness 이 있는 경우 log 변환이 일반적으로 잘 작동한다.

    train_data['SalePrice'] = np.log(train_data['SalePrice'])
    '''
    sns.distplot(train_data['SalePrice'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['SalePrice'], plot=plt)
    plt.show()
    '''
    # 잘 해결된 것을 볼 수 있다.

Check_Normality_label(train_data)


# 'GrLivArea' 에 대해서 살펴본다.

def Check_GrLivArea_Normality(train_data):
    '''
    sns.distplot(train_data['GrLivArea'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['GrLivArea'], plot=plt)
    plt.show()
    '''

    train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])

    '''
    sns.distplot(train_data['GrLivArea'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data['GrLivArea'], plot=plt)
    plt.show()
    '''

Check_GrLivArea_Normality(train_data)

#scatterplot_with_SalePrice('GrLivArea',train_data)# 더이상 특정 모양을 띄지 않는다. 이게 정규화의 힘이다.
# 이는 동분산성 문제를 해결해 준다

# TotalBsmtSF에 대해 살펴본다.

def Check_TotalBsmtSF_Normality(train_data):
    '''
    sns.distplot(train_data['TotalBsmtSF'], fit=norm)
    plt.show()
    fig = plt.figure()
    res  = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
    plt.show()
    '''
    # 데이터가 skewness 를 보인다.
    # 데이터에서 꽤 많은 0 데이터(basement 가 없는 데이터)를 관찰할 수 있다.
    # 이 데이터는 중간에 0 값이 있어서 log 변환을 하지 못한다.

    # log 변환 시킬 수 있는 방법
    # basement 의 유무를 판단할 수 있는 colunm 을 만들고 0이 아닌 데이터에 관해서만 log 변환을 취한다.
    # 이 방법이 basement 가 없는 효과를 잃지 않고 데이터를 변환 시킬 수 있는 방법이다.
    train_data['HasBsmt'] = 0
    train_data.loc[train_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    train_data.loc[train_data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train_data[train_data['HasBsmt'] == 1]['TotalBsmtSF'])

    test_data['HasBsmt'] = 0
    test_data.loc[test_data['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
    test_data.loc[test_data['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(
        test_data[test_data['HasBsmt'] == 1]['TotalBsmtSF'])

    # 0 보다 큰 값들만 그래프로 그린다. 왜냐하면, 0보다 큰 값들만 log 변환시켰기 때문에 0보다 큰 값들에 대해서만 본다.
    '''
    sns.distplot(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
    plt.show()
    fig = plt.figure()
    res = stats.probplot(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
    plt.show()
    '''

Check_TotalBsmtSF_Normality(train_data)

#sns.scatterplot(x=train_data[train_data['TotalBsmtSF']>0]['TotalBsmtSF'], y=train_data[train_data['TotalBsmtSF']>0]['SalePrice'], data=train_data)

# 더이상 특정 모양을 띄지 않는다. 이게 정규화의 힘이다.
# 이는 동분산성 문제를 해결해 준다. (즉, SalePrice 는 TotalBsmtSF 범위에서 동일한 수준의 분산을 나타낸다.)

#plt.show()



# train data Categorical data 수치형 변환


train_data.to_csv('train_data_to_see')
test_data.to_csv('test_data_to_see')

def get_string_data_colunms(train_data):
    string_colunms = []
    for data in train_data.columns:
        if type(train_data[data].iloc[0]) == np.str:
            string_colunms.append(data)
    return string_colunms

def drop_string_same_datas(train_data):
    idx = []
    for colunms in get_string_data_colunms(train_data):
        if len(train_data[colunms].value_counts()) == 1:
            train_data.drop(colunms, axis=1, inplace=True)


drop_string_same_datas(train_data)
ordinal_encoders_colunms = get_string_data_colunms(train_data)


from sklearn.preprocessing import OrdinalEncoder

def string_to_cat(train_data, colunms):
    ordinal_encoder = OrdinalEncoder()
    datas = train_data[[colunms]]
    Transposed = ordinal_encoder.fit_transform(datas)
    train_data[colunms] = Transposed.astype(np.int)


for colunms in ordinal_encoders_colunms:
    string_to_cat(train_data, colunms)


# test_data Categorical data 수치형 변환

drop_string_same_datas(test_data)
ordinal_encoders_colunms = get_string_data_colunms(test_data)
test_data.drop('HasBsmt', axis=1, inplace=True)
print(ordinal_encoders_colunms)

for colunms in ordinal_encoders_colunms:
    string_to_cat(test_data, colunms)


print(len(get_string_data_colunms(train_data)))
print(len(get_string_data_colunms(test_data)))


def column_test(train_data, test_data):
    for col in train_data:
        if col not in test_data:
            print(col)

column_test(train_data, test_data)


# model prediction

# LinearRegression, SVR, RandomForest, Xgboost, LightGBM

# split label, train data

def split_label(train_data):
    label = train_data[['SalePrice']]
    train = train_data.iloc[:,:-2]
    return train, label

x_train, y_train = split_label(train_data)
print(x_train)
print(y_train)
column_test(train_data, test_data)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

def Models(x_train, y_train):

    model_list = [(LinearRegression(n_jobs=-1),"LinearRegression") ,
                  (SVR(kernel='rbf',gamma=0.1, C=1,epsilon=0.1),"SVR"),
                  (RandomForestRegressor(n_estimators=1000, max_leaf_nodes=16, n_jobs=-1),"RandomForestRegressor"),
                  (XGBRegressor(objective='reg:squarederror',n_estimators=1000, learning_rate=0.1,n_jobs=-1),"XGBRegressor"),
                  (AdaBoostRegressor(base_estimator=Ridge(alpha=2), n_estimators=1000, learning_rate=0.5), "AdaBoostRegressor"),
                  (GradientBoostingRegressor(max_depth=2, n_estimators=1100,),"GBM"),
                  (LGBMRegressor(boosting_type='gbdt',n_estimators=200, n_jobs=-1),"LGBM"),
                  (Ridge(alpha=1), "Ridge")
                  ]

    for model, name in model_list:
        train_model = model
        train_model.fit(x_train, y_train)
        rmse_score = cross_val_score(estimator=model, X=x_train, y=y_train, scoring="neg_mean_squared_error", cv=10)
        print(name,"RMSE value")
        print(np.sqrt(-rmse_score).mean())
        print()

Models(x_train, y_train.values.flatten())


# 하이퍼 파라미터 튜닝

'''
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'max_depth': [2,3,4], 'n_estimators':[1000,1100,1200,1300],'learning_rate':[0.1,0.2,0.3]}
]

gbm = GradientBoostingRegressor(max_depth=2, n_estimators=1100)

grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=10, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(x_train, y_train.values.ravel())
print(np.sqrt(-grid_search.best_score_).mean())
print(grid_search.best_estimator_)
print(grid_search.best_params_)
'''

'''
0.11589065003265452
GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.1, loss='ls', max_depth=2,
                          max_features=None, max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1100,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=1.0, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
{'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 1100}

'''


# 모델 선정
'''
model1 = XGBRegressor(objective='reg:squarederror',n_estimators=900, learning_rate=0.1,n_jobs=-1)
model1.fit(x_train, y_train)
y_predict = model1.predict(test_data)

result_df = pd.DataFrame({
    'Id': test_indexes.to_numpy(),
    'SalePrice': np.exp(y_predict)
})
result_df = result_df.set_index('Id')
result_df.to_csv("submission.csv")

print(result_df)
'''

model1 = GradientBoostingRegressor(max_depth=2, n_estimators=1100,learning_rate=0.1)
model1.fit(x_train, y_train.values.ravel())
y_predict = model1.predict(test_data)

result_df = pd.DataFrame({
    'Id': test_indexes.to_numpy(),
    'SalePrice': np.exp(y_predict)
})
result_df = result_df.set_index('Id')
result_df.to_csv("submission.csv")

print(result_df)



# 신경망

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

print(x_train.shape)
print(x_test.shape)



import tensorflow as tf
import os
import datetime

# 얕은 신경망
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=25, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=25, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=10, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=5, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=3, activation='elu', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dense(units=1))

s = 1500 * x_train.shape[0] // 32
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)
model.compile(optimizer=optimizer, loss='mse', metrics='mse')


model.fit(x_train, y_train, batch_size=32, epochs=3000, validation_split=0.2, shuffle=True)
evaluation = model.evaluate(x_test, y_test)
print(np.sqrt(evaluation))

```



















