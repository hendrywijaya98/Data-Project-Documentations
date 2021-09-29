# Data Project Starter
```
# importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Data Inspection
### Reading data and Creating DataFrame
```
# reading data
PATH = 'https://raw.githubusercontent.com/rmpbastos/data_sets/main/kaggle_housing/house_df.csv'
PATH = 'path_to_data.csv'
df = pd.read_csv(PATH)

# creating dataframe
df = pd.DataFrame(mydict)

# check the type of the DataFrame
type(df)

# read dataframe with set index to certain column
df = pd.read_csv(PATH, index_col='column')

# set index into one column
df.set_index('column', inplace=True)
df.index
```
### Examining DataFrame
```
# check from the first rows dataset (first 5 rows in default)
df.head()

# check from the last rows dataset (last 5 rows in default)
df.tail()

# check dataset info, including each colum data type and null values
df.info()

# check dataframe describe
df.describe()

# check dataset dimension, in this case for column and rows
df.shape

# shape[0] for rows and shape[1] for columns
print(f'{df.shape[0]} rows and {df.shape[1]} column')

# count each uniqe value in categorical column
df['Column'].value_counts()

# checking data type from each column 
# example data type: float, int, string, bool
df.dtypes
```

# Data Manipulation
## Rows and Columns
```
# check data frame column
df.columns

# check column from the first rows (five in default)
df['column name'].head()

# check column data type
type(df['column name'])

# selecting a column by a name
df.col_name
df['column name']

# selecting multiple column by name
df[['column 1', 'column 2']]

# selecting column value based on condition
df[conditions]['column name']
```
## Adding Columns
```
# make a copy to dataset
df_copy = df.copy()
# add new column with only 1 values in a single series
df_copy['Sold'] = 'N'
```
## Adding Rows
```
data_to_append = {'LotArea': [9500, 15000],
                  'Steet': ['Pave', 'Gravel'],
                  'Neighborhood': ['Downtown', 'Downtown'],
                  'HouseStyle': ['2Story', '1Story'],
                  'YearBuilt': [2021, 2019],
                  'CentralAir': ['Y', 'N'],
                  'Bedroom': [5, 4],
                  'Fireplaces': [1, 0],
                  'GarageType': ['Attchd', 'Attchd'],
                  'GarageYrBlt': [2021, 2019],
                  'GarageArea': [300, 250],
                  'PoolArea': [0, 0],
                  'PoolQC': ['G', 'G'],
                  'Fence': ['G', 'G'],
                  'SalePrice': [250000, 195000],
                  'Sold': ['Y', 'Y']}

df_to_append = pd.DataFrame(data_to_append)
df_to_append
```
Append the 2-row DataFrame above to df_copy
```
df_copy = df_copy.append(df_to_append, ignore_index=True)
```
Renaming columns
```
df.rename(columns={'BedroomAbvGr': 'Bedroom'}, inplace=True)
```
Date Column Transformation
```
# change data type column to datetime
df['date'] = pd.to_datetime(df['date'])

# date transformation to year/month/day single column
df['year'] = df['dt_iso'].dt.year
df['month'] = df['dt_iso'].dt.month
df['day'] = df['dt_iso'].dt.day
df['hour'] = df['dt_iso'].dt.hour

# function to transform date column
def DateTransform(data, column):
	# change data type column to datetime
	data[column] = pd.to_datetime(data[column])
	
	# date transformation to year/month/day single column
	data['year'] = data[column].dt.year
	data['month'] = data[column].dt.month
	data['day'] = data[column].dt.day
	data['hour'] = data[column].dt.hour
		
	return data['year'], data['month'], data['day'], data['hour']
```
String to Numeric Transformation
```
# change column from decimal to numeric 
df['CAD_USD'] = pd.to_numeric(df.CAD_USD, errors='coerce')
df.dropna(inplace=True)

# change column from string (object) to float
for row in range(len(df)):
    if 'MWp' in df.loc[row, 'Peak_capacity']:
        df.loc[row,"New_Col"] = float(df.loc[row, 'Peak_capacity']
                                      .split('MWp')[0])
    else:
        df.loc[row,"New_Col"] = float(df.loc[row, 'Peak_capacity'])
```
### Selecting Column by Index using loc and iloc
**loc** is used to **accessing rows** and **columns by label** or **index** based on boolean array
```
# access rows with 1000 value as index
df.loc[1000]

# access rows and multiple column with 1000 value as index
df.loc[1000, ['column 1', 'column 2']]

# selecting multiple column by name as index
df.loc[:['column 1', 'column 2']]

# select column with its value condition
df.loc[df['column 1'] >= <value_condition>]
```
**iloc** is used to **accessing rows** and **columns** data **based on integer location** or a **boolean array**
```
# selecting data contained in first rows and first column
df.iloc[0,0]

# selecting column by index
df.iloc[:, [0:2]]

# select an entire column in 10th row position
df.iloc[10,: ]

# select an entire column in last column
df.iloc[:,-1]

# select multiple rows and column by index
df.iloc[8:12, 2:5]
```
### Filtering Column
```
# filter colom berdasarkan kondisi tertentu
# copy() untuk duplikasi
df_outlier3 = df[(df[‘Length’]> 215)&(df[‘Right’]> 130)&(df[‘Left’]> 130)&(df[‘Bottom’]> 10)].copy()

print(Counter(df_outlier3[‘conterfeit’]))
```
# Handling Missing Value
## Checking Missing Value
```
# check column value that are null
df.isnull()

# check null value from each column
df.isnull().sum()

# check the percentage of null value from each column
df.isnull().sum / df.shape[0]

# get percentage of a couple of columns with missing value
for column in df.columns:
    if df[column].isnull().sum() > 0:
        print(column, ': {:.2%}'.format(df[column].isnull().sum() / df[column].shape[0]))
```
## Removing Missing Value
```
# add inplace = True to do save over current dataframe
# remove a single column
df.drop(labels='column', axis=1, inplace=True)
# remove rows with missing value from single column in subset
df.dropna(subset=['column'], axis=0, inplace=True)

# remove multiple column
df.drop(['column 1', 'column 2'], axis = 1)
```
## Filtering Missing Value
```
# fill missing value with certain value
df['column'].fillna(value='column value', inplace=True)

# fill missing value from median calculation result
column_median = df['column'].median()
df.fillna({'column' : column_median}, inplace=True)
```
# Handling Duplicate Value
```
df[df.duplicated()].count()

# inplace mean in real dataframe 
df.drop_duplicates(inplace=True)
# last means last time a row of data added
df.drop_duplicates(keep='last', inplace=True)
# first means first time a row of data added
df.drop_duplicates(keep='first', inplace=True)
# drop duplicates to a specific column 
df.drop_duplicates(subset = ['column name'], inplace=True)
# drop duplicates for multiple specific columns 
df.drop_duplicates(subset = ['column 1', 'column 2'], inplace=True)
```
# Handling Outlier Value
### Using Isolation Forest
metode yang secara random memilih kolom dan value nya untuk memisahkan bagian data, pada data yang kompleks karena banyak kolom dan numerical value multi modal

metode yang secara random memilih kolom dan value nya untuk memisahkan bagian data, pada data yang kompleks karena banyak kolom dan numerical value multi modal

validate our predictions karena datanya terdapat counterfeit labels
```
# import library
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import numpy as np

# fit to the length, left, right, bottom, top and diagonal column as features
X = df[[‘Length’, ‘Left’, ‘Right’, ‘Bottom’, ‘Top’, ‘Diagonal’]]
# conterfeit as label
y = df[‘conterfeit’]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# define the model to fit train data
clf = IsolationForest(random_state=0)
clf.fit(X_train)
y_pred = clf.predict(X_test)

# predict and evaluate test data
pred = pd.DataFrame({‘pred’: y_pred})
pred[‘y_pred’] = np.where(pred[‘pred’] == -1, 1, 0)
y_pred = pred[‘y_pred’]
# display the precision score
print(“Precision:”, precision_score(y_test, y_pred))
```
### Using One Class SVM
Another unsupervised machine learning technique that is useful for high dimensional and large data sets
```
# import OneClassSVM from sklearn.svm
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import numpy as np

# define the model
clf_svm = OneClassSVM(gamma=’auto’)
clf_svm.fit(X_train)

y_pred_svm = clf_svm.predict(X_test)
pred[‘svm’] = y_pred_svm
pred[‘svm_pred’] = np.where(pred[‘svm’] == -1, 1, 0)
y_pred_svm = pred[‘svm_pred’]

# checking precision score before outlier handling
print(“SVM Precision:”, precision_score(y_test, y_pred_svm))
```


# Data Visualization
using matplotlib and seborn
## Histogram
```
# plot the distribution using histogram argument at kind
# x-axis contain bins that devide the values into intervals
# y-axis contain frequency
df['column'].plot(kind='hist')

# plot histogram for all numeric columns
df.hist
```
## Scatter Plot
```
# plot scatter to check relation betwen column 1 and column 2
df.plot(x='column 1', y='column 2', kind='scatter')
```
## Box Plot
```
# plot the boxplot to detect outliers
# x-axis sebagai nama kolom yang dicari outliernya
def boxplot(column):
		# column menerima argument, kemudian dimasukan ke fstring
		sns.boxplot(data=df, x=df[f'{column}'])
		plt.title(f'Boxplot of Swiss Banknote {column}')
		plt.show()

# jalankan fungsi boxplot
boxplot('Column 1')
boxplot('Column 2')
boxplot('Column 3')
```
## Bar Plot
special snippet barplot visualization with seaborn 
```
# Set up Matplotlib Figure
f, ax1 = plt.subplots(1, 1, figsize=(15, 8))

# Set up the data for the visualization
data_for_plot = df.groupby(by='Country').sum()
data_for_plot.reset_index(inplace=True)
data_for_plot.sort_values(by='Area_Acres', inplace=True)
values_on_x_axis = 'Country'
values_on_y_axis = 'Area_Acres'

# Setup actual data visualization in Seaborn
sns.barplot(x=values_on_x_axis,y=values_on_y_axis,
            data=data_for_plot,
            ax = ax1,
            palette="Blues")

# Set additional elements of the Visualization
plt.title("Total Area per country", fontsize=20)
plt.xlabel("Country", fontsize=15)
plt.ylabel("Total Area in Acres", fontsize=15)
```
# Saving to File
```
# save dataset output as csv file 
df.to_csv('Data Output.csv')

# specify a directory to save a csv file
df.to_csv('C:/Users/username/Documents/Data Output.csv')
```
