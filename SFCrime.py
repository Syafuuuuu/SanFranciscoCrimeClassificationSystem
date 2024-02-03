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

print("Category:")
print(train['Category'].nunique())
print(train['Category'].unique())

# print("DayOfWeek:")
# print(train['DayOfWeek'].unique())

print("PdDistrict:")
print(train['PdDistrict'].unique())

# Create scatter map
# fig = px.scatter_geo(train, lat='Y', lon='X', title='Crime Incidents in San Fransisco', projection="natural earth")
# fig.show()

#----------Remove duplicates-----------
print("Duplicate count before:")
print(train.count().duplicated())
print(train.count())
train.drop_duplicates(inplace=True)

print("Duplicate count after:")
print(train.count().duplicated())
print(train.count())

#region---------- Remove Incorrect Coordinates ------------
train.drop_duplicates(inplace=True)
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
test.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])
    test.loc[test['PdDistrict'] == district, ['X', 'Y']] = imp.transform(
        test.loc[test['PdDistrict'] == district, ['X', 'Y']])

# print(train.head())

#----------Remove the outlier data points------
# train = train.drop(['Descript','Resolution','Address'], axis='columns')
# test = test.drop(['Id'], axis='columns')
# print(train.head())

#endregion

#region---------| Look at how the distribution is among days of the week |-------
data = train.groupby('DayOfWeek').count().iloc[:, 0]
data = data.reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
    'Sunday'
])

plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        x=data.index, y=(data.values / data.values.sum()) * 100,
        orient='v',
        palette=cm.ScalarMappable(cmap='Reds').to_rgba(data.values))

plt.title('Incidents per Weekday', fontdict={'fontsize': 16})
plt.xlabel('Weekday')
plt.ylabel('Incidents (%)')

plt.show()
#endregion

#region-------| Look at what crime is most frequent |----------
data = train.groupby('Category').count().iloc[:, 0].sort_values(
    ascending=False)
data = data.reindex(np.append(np.delete(data.index, 1), 'OTHER OFFENSES'))

plt.figure(figsize=(10, 10))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        x=(data.values / data.values.sum()) * 100,
        y=data.index,
        orient='h',
        palette="Reds_r")

plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})
plt.xlabel('Incidents (%)')

plt.show()
#endregion

#region -------| transforming Dates |---------
def dateEncoder(x):
    x['Dates'] = pd.to_datetime(x['Dates'], errors='coerce')
    x['Month'] = x.Dates.dt.month
    x['Hour'] = x.Dates.dt.hour

dateEncoder(train)
dateEncoder(test)
#endregion

#region----------| check distribution by month |------------
data = train.groupby('Month').count().iloc[:, 0]

plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        x=data.index, y=data.values,
        orient='v',
        palette=cm.ScalarMappable(cmap='Blues').to_rgba(data.values))

plt.title('Incidents per Month', fontdict={'fontsize': 16})
plt.xlabel('Month')
plt.ylabel('Incidents (%)')

plt.show()
#endregion

#region----------| check distribution by hour |------------
data = train.groupby('Hour').count().iloc[:, 0]

plt.figure(figsize=(10, 5))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        x=data.index, y=data.values,
        orient='v',
        palette=cm.ScalarMappable(cmap='Greens').to_rgba(data.values))

plt.title('Incidents per Hour', fontdict={'fontsize': 16})
plt.xlabel('Hour')
plt.ylabel('Incidents (%)')

plt.show()
#endregion

#----------| Check by the hour but with respect to the category |-----------
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")

# Group the data by hour and category and count the occurrences
hourly_category_counts = train.groupby(['Hour', 'Category']).size().reset_index(name='Count')

# Plot the lineplot
ax = sns.lineplot(x='Hour', y='Count', hue='Category', data=hourly_category_counts, palette='viridis')

# Set labels and title
plt.title('Incidents per Hour with Respect to Category', fontsize=16)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Incident Count', fontsize=12)
plt.xticks(rotation=45)

# Adjust legend
ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


# Get unique categories
categories = train['Category'].unique()

# Create a line graph for each category
plt.figure(figsize=(16, 12))
for category in categories:
    sns.set_style("whitegrid")
    
    # Filter data for the current category
    category_data = hourly_category_counts[hourly_category_counts['Category'] == category]
    
    # Plot the lineplot
    ax = sns.lineplot(x='Hour', y='Count', data=category_data, color='blue')
    
    # Set labels and title
    plt.title(f'Incidents per Hour for Category: {category}', fontsize=16)
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Incident Count', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.show()
# plt.show()



print(train)
#-------| Encoding our data |---------

print("--------------------------------------------")
print("-------------- After Encoding --------------")
print("--------------------------------------------")

print(test.head())
print(train_clean.head())

train_clean_corr = train_clean[['Category','DayOfWeek','PdDistrict','X','Y', 'Month', 'Hour']]
print(train_clean_corr.corr())
sns.heatmap(train_clean_corr.corr(), annot=True)

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




#Calculate the skew
skew = train_clean_corr.skew()
print(skew)

# data_corr_map = sns.heatmap(train_clean_corr.corr(), annot=True)
plt.show()

