import pandas as pd
#from shapely.geometry import  Point
#import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import cm
import urllib.request
import shutil
import zipfile
import os
import re
#import contextily as ctx
#import geoplot as gplt
#import lightgbm as lgb
#import eli5
# from eli5.sklearn import PermutationImportance
# from lightgbm import LGBMClassifier
# from matplotlib import pyplot as plt
# from pdpbox import pdp, get_dataset, info_plots
# import shap

train = pd.read_csv(r'C:\Users\User\Desktop\SEM 5\Pattern Rec\Project\train.csv')
test = pd.read_csv(r'C:\Users\User\Desktop\SEM 5\Pattern Rec\Project\test.csv')

# print('First date: ', str(train.Dates.describe()['first']))
# print('Last date: ', str(train.Dates.describe()['last']))
# print('Test data shape ', train.shape)

# print("Train Head:")
# print(train.head(10))

# print("Category:")
# print(train['Category'].nunique())
# print(train['Category'].unique())

# print("DayOfWeek:")
# print(train['DayOfWeek'].unique())

# print("PdDistrict:")
# print(train['PdDistrict'].unique())

# Create scatter map
# fig = px.scatter_geo(train, lat='Y', lon='X', title='Crime Incidents in San Fransisco', projection="natural earth")
# fig.show()

print("Duplicate count before:")
print(train.count().duplicated())
print(train.count())
train.drop_duplicates(inplace=True)

print("Duplicate count after:")
print(train.count().duplicated())
print(train.count())

train_clean = train.drop(['Descript','Resolution','Address'], axis='columns')
test = test.drop(['Id'], axis='columns')
print(train_clean.head())

print("--------------------------------------------")
print("-------------- Before Encoding --------------")
print("--------------------------------------------")

print(train_clean.head())
print(test.head())

#-------| Encoding our data |---------

#-------| Encoding Category |---------
uniqueCat = train_clean['Category'].unique()

cat_dict = {}
count = 1
for data in uniqueCat:
    cat_dict[data] = count
    count+=1

train_clean["Category"] = train_clean["Category"].replace(cat_dict)


#-------| Encoding Weekdays |---------
week_dict = {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}

train_clean["DayOfWeek"] = train_clean["DayOfWeek"].replace(week_dict)
test["DayOfWeek"] = test["DayOfWeek"].replace(week_dict)


#-------| Encoding Weekdays |---------
district = train_clean["PdDistrict"].unique()
district_dict = {}
count = 1
for data in district:
    district_dict[data] = count
    count+=1 

train_clean["PdDistrict"] = train_clean["PdDistrict"].replace(district_dict)
test["PdDistrict"] = test["PdDistrict"].replace(district_dict)

#-------| Encoding Dates |---------
def dateEncoder(x):
    x['Dates'] = pd.to_datetime(x['Dates'], errors='coerce')
    x['Month'] = x.Dates.dt.month
    x['Hour'] = x.Dates.dt.hour

dateEncoder(train_clean)
dateEncoder(test)

print("--------------------------------------------")
print("-------------- After Encoding --------------")
print("--------------------------------------------")

print(train_clean.head())
print(test.head())

train_clean_corr = train_clean[['Category','DayOfWeek','PdDistrict','X','Y', 'Month', 'Hour']]
print(train_clean_corr.corr())
sns.heatmap(train_clean_corr.corr(), annot=True)

#Calculate the skew
skew = train_clean_corr.skew()
print(skew)

# data_corr_map = sns.heatmap(train_clean_corr.corr(), annot=True)
plt.show()

