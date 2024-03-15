# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:11:50 2024

@author: Ashish Chincholikar
Support Vector Machine(SVM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


forest = pd.read_csv("C:/SVM/DataSets/forestfires.csv")
forest.dtypes
forest.head()

#forest['month'] = forest['month'].astype(int)
# EDA
forest.shape

plt.figure( 1 , figsize = (16,10))
sns.countplot(data = forest , x = forest.day)
# AUG and SEPT has higest value
sns.countplot(data = forest , x = forest.day)
# FRIDAY SUNDAY AND SATURDAY has higest value

sns.distplot(forest.FFMC)
#data isnormal and slight left skewed

sns.boxplot(forest.FFMC)
#There are several outliers

sns.distplot(forest.DC)
# data is normal and slight left skewed
sns.boxplot(forest.DC)

sns.distplot(forest.RH)
# data is normal and slight left shewed
sns.boxplot(forest.RH)
#there are outliers

sns.distplot(forest.wind)
#data is normal and slight right skewed
sns.boxplot(forest.wind)
#there are outliers

sns.distplot(forest.rain)
#data is normal
sns.boxplot(forest.rain)
#there are outliers

sns.distplot(forest.area)
#data is normal
sns.boxplot(forest.data)
#there are outliers

#now let us check the highest fire in KM?
forest.sort_values(by = "area" , ascending = False).head(5)
highest_fire_area = forest.sort_values(by = "area" , ascending = True)

plt.figure(figsize = (8,6))
plt.title("Temperature vs area of fire")
plt.bar(highest_fire_area['temp'] , highest_fire_area['area'])
plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()

#once the fire starts , almost 1000+ sq area's
#temperature goes beyond 25 and around 750km area is facing temp 30+
#now let us check the highest rain in the forest
highest_rain = forest.sort_values(by = 'rain' , ascending = False)[['month' , 'day' , 'rain']].head(5)
highest_rain


#highest rain observed in the month of august
# let us check highest and lowest temperature in month and 
highest_temp = forest.sort_values(by = "temp" , ascending = False)[['month' , 'day' , 'rain']].head(5)

lowest_temp = forest.sort_values(by = 'temp' , ascending = True)[['month' , 'day' , 'rain']].head(5)
print("highest temperature : " , highest_temp)
#highest temperature observed in august
print("lowest temperature : " , lowest_temp)
#lowest temperature observed in December

forest.isna().sum()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
forest.month = labelencoder.fit_transform(forest.month)
forest.day = labelencoder.fit_transform(forest.day)
forest.size_category = labelencoder.fit_transform(forest.size_category)

forest.dtypes
from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['month'])
df_t = winsor.fit_transform(forest[['month']])
sns.boxplot(df_t.month)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['FFMC'])
df_t = winsor.fit_transform(forest[['FFMC']])
sns.boxplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['RH'])
df_t = winsor.fit_transform(forest[['RH']])
sns.boxplot(df_t.RH)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['wind'])
df_t = winsor.fit_transform(forest[['wind']])
sns.boxplot(df_t.wind)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['rain'])
df_t = winsor.fit_transform(forest[['rain']])
sns.boxplot(df_t.rain)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['area'])
df_t = winsor.fit_transform(forest[['area']])
sns.boxplot(df_t.area)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['DC'])
df_t = winsor.fit_transform(forest[['DC']])
sns.boxplot(df_t.DC)

"""

from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method = 'iqr' , tail = 'both' , fold = 1.5 ,variables=['month'])
df_t = winsor.fit_transform(forest[['month']])
sns.boxplot(df_t.month)


apply this code to all the columns to remove the outlier values """


tc = forest.corr()
tc
fig , ax = plt.subplots()
fig.set_size_inches(200 , 10)
sns.heatmap(tc , annot = True)
#all the variable are slightly correlated with size_category

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train , test = train_test_split(forest,  test_size = 0.3)
train_X = train.iloc[: , :30]
train_y = train.iloc[: , 30]
test_X = test.iloc[:  , :30]
test_y = test.iloc[ : , 30]

#kernel linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X , train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear == test_y)
#RBF
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X , train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf == test_y)














