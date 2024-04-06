import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

url = "C:\\Users\\denis\\Desktop\\ML Projects\\House-Price-Prediction\\HousePricePrediction.xlsx"
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
# unq_val = []
# Determining how many unique values are in each object column
# pd.size counts the total number of values both valid and NaN. pd.count counts the total number of values of only valid entries.
# for col in obj_col:
#     unq_val.append(data[col].unique().size)
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

# i = 1
# for col in obj_col:
#     y = data[col].value_counts()
#     plt.subplot(1, 4, i)
#     plt.xticks(rotation = 90)
#     sns.barplot(x = list(y.index), y = y)
#     i += 1
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


# -----Machine Learning-----

# Splitting data to training and testing
x = final_data.drop(['SalePrice'], axis = 1) # Seperate the dataset from the label column
y = final_data['SalePrice'] # Separate the label column from the rest of the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# SVM
svm_reg = svm.SVR() # Initialize SVM (Regressor)
svm_reg.fit(x_train, y_train) # Train model
ypred = svm_reg.predict(x_test) # Test model
print(mean_absolute_percentage_error(y_true = y_test, y_pred = ypred))
# Results are: 0.1870512931870423 ~ 18.71% error.

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators = 10) # Initialize Random Forest Regressor
rf_reg.fit(x_train, y_train) # Train model
ypred = rf_reg.predict(x_test) # Test model
print(mean_absolute_percentage_error(y_true = y_test, y_pred = ypred))
# Results are: 0.19331228305022413 ~ 19.3% error

# Linear Regression
lin_reg = LinearRegression() # Initialize Linear Regression
lin_reg.fit(x_train, y_train) # Train model
ypred = lin_reg.predict(x_test) # Test model
print(mean_absolute_percentage_error(y_true = y_test, y_pred = ypred))
# Results are: 0.1874168384159985 ~ 18.74% error

# Bonus: CatBoost Classifier, by YandeX (open source)
catb_reg = CatBoostRegressor() # Initialize CatBoost Classifier Regression
catb_reg.fit(x_train, y_train) # Train model
ypred = catb_reg.predict(x_test)
print(mean_absolute_percentage_error(y_true = y_test, y_pred = ypred))
print(r2_score(y_true = y_test, y_pred = ypred))
# Results are: 0.18178925297425216 ~ 18.17% error
# R2 score: 0.38351169878113034

# Results somehow show that CatBoost was the ideal and most accurate model as it had the least error with 0.18178925297425216.

# -----Machine Learning-----
