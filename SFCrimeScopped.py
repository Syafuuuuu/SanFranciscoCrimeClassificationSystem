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
    
#endregion

#region------------| Scope down to top 5 crimes |---------------
print("Values before scoping down")
print(train.head())
train = train[train['Category'].isin(['LARCENY/THEFT', 'NON-CRIMINAL', 'ASSAULT', 'DRUG/NARCOTIC', 'VEHICLE THEFT'])]
print("Values after scoping down")
print(train.head())

#saving said scoped down dataset
# train.to_csv("Scoped_train.csv")

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

# plt.show()
#endregion

#region-------| Look at what crime is most frequent |----------
data = train.groupby('Category').count().iloc[:, 0].sort_values(
    ascending=False)
data = data.reindex()

plt.figure(figsize=(10, 10))
with sns.axes_style("whitegrid"):
    ax = sns.barplot(
        x=(data.values / data.values.sum()) * 100,
        y=data.index,
        orient='h',
        palette="Reds_r")

plt.title('Incidents per Crime Category', fontdict={'fontsize': 16})
plt.xlabel('Incidents (%)')

# plt.show()
#endregion

#region -------| transforming Dates |---------
def dateEncoder(x):
    x['Dates'] = pd.to_datetime(x['Dates'], errors='coerce')
    x['Month'] = x.Dates.dt.month
    x['Hour'] = x.Dates.dt.hour
    x['Minute'] = x.Dates.dt.minute

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

# plt.show()
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

# plt.show()
#endregion

#region----------| Check by the hour but with respect to the category |-----------
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

# plt.show()


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
    
    # plt.show()
# # plt.show()

#endregion



print(train)
#-------| Encoding our data |---------

print("--------------------------------------------")
print("-------------- After Encoding --------------")
print("--------------------------------------------")

#-------| Encoding Category |---------
uniqueCat = train['Category'].unique()

cat_dict = {}
count = 1
for data in uniqueCat:
    cat_dict[data] = count
    count+=1

train["Category"] = train["Category"].replace(cat_dict)


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

train["DayOfWeek"] = train["DayOfWeek"].replace(week_dict)
test["DayOfWeek"] = test["DayOfWeek"].replace(week_dict)


#-------| Encoding Districts |---------
district = train["PdDistrict"].unique()
district_dict = {}
count = 1
for data in district:
    district_dict[data] = count
    count+=1 

train["PdDistrict"] = train["PdDistrict"].replace(district_dict)
test["PdDistrict"] = test["PdDistrict"].replace(district_dict)


train_corr = train[['Category','DayOfWeek','PdDistrict','X','Y', 'Month', 'Hour', 'Minute']]
print(train_corr.corr())
sns.heatmap(train_corr.corr(), annot=True)

#Calculate the skew
skew = train_corr.skew()
print(skew)

# data_corr_map = sns.heatmap(train_clean_corr.corr(), annot=True)
# plt.show()


#-------------| Drop unwated features |-------------
train = train.drop(['Dates','Descript','Resolution','Address'], axis=1)
test = test.drop(['Id','Dates','Address'], axis=1)

print(train)
print(test)
target = 'Category'

x = train.drop([target], axis=1)
y = pd.DataFrame()
y.loc[:,target] = train.loc[:,target]

# x.to_csv("Train_Features.csv")
# y.to_csv("Train_Target.csv")

def  model_training(x,y):
    #----------{Split Data}-------------
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

    # ------------| normalization - feature standard scaling |----------- 
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    st_x= MinMaxScaler()    
    x_train= st_x.fit_transform(x_train)
    x_test = st_x.fit_transform(x_test)

    # Reshape y_train and y_test
    y_train = y_train.values.flatten()
    y_test = y_test.values.flatten()

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB

    # k-NN: k=3
    k = 10
    KNNclassifier = KNeighborsClassifier(n_neighbors=k,metric='minkowski', p=2)
    KNNclassifier.fit(x_train, y_train)
    knn_accuracy = KNNclassifier.score(x_test, y_test)

    #Log Reg
    LogReg = LogisticRegression(verbose=2)
    LogReg.fit(x_train,y_train)
    log_reg_accuracy = LogReg.score(x_test, y_test)

    #Random Forest
    rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',random_state =7, verbose=10)
    rfc.fit(x_train, y_train)
    rfc_acc = rfc.score(x_test, y_test)


    #MLP Neural Nets
    mlp = MLPClassifier(solver='adam', activation='relu', alpha=1e-05, tol = 1e-04, hidden_layer_sizes=(20,),random_state=1, max_iter = 1000, verbose=2)
    mlp.fit(x_train, y_train)
    mlp_acc = mlp.score(x_test, y_test)

    # Multinomial Naive Bayes
    MLB = BernoulliNB()
    MLB.fit(x_train, y_train)
    MLB_acc = MLB.score(x_train, y_train)

    # Outputs and Scores
    print("KNN (k=3) Accuracy:", knn_accuracy)
    print("Logistic Regression Accuracy:", log_reg_accuracy)
    print("RF Acc: ", rfc_acc)
    print("MLP Acc: ", mlp_acc)
    print("BLB: ", MLB_acc) 
    # print(x.head())
    # print(rfc.feature_importances_)

model_training(x,y)
