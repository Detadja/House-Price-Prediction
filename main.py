import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

url = "C:\\Users\\User\\Desktop\ML Projects\\House-Price-Prediction\\HousePricePrediction.xlsx"
data = pd.read_excel(url)
# print(data.head())
# print(data.shape)
# print(data.dtypes)
# print(data.dtypes.unique()) # There are 3 types: int, float and object only.

# -----Preprocessing, cleaning, EDA, and One Hot Encoder-----

# Each of these list out all of the columns and assign T or F in each category (int, float, object)
# obj[obj] for e.g., will list out all of the columns that are assigned T in each category (int, float, object)
obj = (data.dtypes == 'object')
integer = (data.dtypes == 'int')
decimal = (data.dtypes == 'float')

# Lists out all the columns that are assigned True in each category (int, float, object)
# Grabs all the columns assigned as True for the attribute 'object' (obj[obj], integer[integer], decimal[decimal]),
# -index them and put them in a list.
obj_col = list(obj[obj].index)
int_col = list(integer[integer].index)
float_col = list(decimal[decimal].index)
# print(obj_col) # 4 cols
# print(int_col) # 6 cols
# print(float_col) # 3 cols

# plt.figure(figsize = (12, 6))
# # corr() is used to find the pairwise correlation of all columns in the Pandas Dataframe 
# sns.heatmap(data.corr(), cmap = 'BrBG', fmt = '.2f', linewidths = 2, annot = True)
# plt.show()


# Analyzing different categorical features
unq_val = []
# Determining how many unique values are in each object column
# pd.size counts the total number of values both valid and NaN. pd.count counts the total number of values of only valid entries.
for col in obj_col:
    unq_val.append(data[col].unique().size)
# plt.figure(figsize = (10, 6))
# plt.title('No. Unique values of Categorical Features')
# plt.xticks(rotation = 90)
# sns.barplot(x = obj_col, y = unq_val)
# plt.show()
# MSZoning has 6 unique values
# LotConfig has 5 unique values
# BldgType has 5 unique values
# Exterior1st has 16 unique values

# Determining how many counts of each unique value in each object column
# plt.figure(figsize = (18, 36))
# plt.title('Categorical Features: Distribution')
# plt.xticks(rotation = 90)

i = 1
for col in obj_col:
    y = data[col].value_counts()
    plt.subplot(1, 4, i)
    plt.xticks(rotation = 90)
    sns.barplot(x = list(y.index), y = y)
    i += 1
# plt.show()


# Cleaning data
data.drop(['Id'], axis = 1, inplace = True) # Dropping Id column
# Fills the missing entries in SalePrice column with the average of the remaining entries.
data['SalePrice'] = data['SalePrice'].fillna(data['SalePrice'].mean())
clean_data = data.dropna() # Drop records with null values
# print(clean_data.isnull().sum()) # There are no entries with null values


# One Hot encoder
obj_clean = (clean_data.dtypes == 'object')
# Grabs all the columns assigned as True for the attribute 'object' (obj_clean[obj_clean]), index them and put it in a list, from the-
# -cleaned dataset.
clean_obj_col = list(obj_clean[obj_clean].index)

# Applying OHE to the whole list
ohe = OneHotEncoder(sparse = False) # Initialize
# Fills the df with ohe notation values (0.0 or 1.0) in the rows for each column.
ohe_col = pd.DataFrame(ohe.fit_transform(clean_data[clean_obj_col]))
ohe_col.index = clean_data.index # Places the correct index in the ohe_col, based on clean_data.index
ohe_col.columns = ohe.get_feature_names_out() # Places the correct labels (column names) for each column
# print(ohe_col)

final_data = clean_data.drop(clean_obj_col, axis = 1) # Drop all 'object' columns from cleaned dataset
# Concatenate final_data with ohe_col horizontally to the right (default is axis = 0, where it will concatenate below).
final_data = pd.concat([final_data, ohe_col], axis = 1)

# -----Preprocessing, cleaning, EDA, and One Hot Encoder END-----


