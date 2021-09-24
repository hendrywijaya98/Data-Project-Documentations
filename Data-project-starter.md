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
