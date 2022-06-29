from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,Normalizer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.data_helper import *
from helpers.dataPreProcessing import *
from helpers.machineLearning import *

train = pd.read_csv("datas/Train.csv")
train_df = train.copy()
test = pd.read_csv("datas/Test.csv")
test_df = test.copy()
df = pd.concat([train_df, test_df]) # default: axis=0
# yinelenen degerlere sahip olan satırları kaldırmak için kullandım.
df = df[~df.index.duplicated()]


# 1-DATA UNDERSTANDING

print(dataUnderstand(df))
# Item_Outlet_Sales, Item_MRP(small), Item_Visibility => Outlier value
for col in df.columns:
    print(dataUnderstand(df).col_Corr(col))

# Columns Numeric, Categoric and Numeric but Categoric olarak ayırma
categoric_cols, numeric_cols, categoric_but_numeric = dataUnderstand(
    df).features()
print("Categoric Columns: ", categoric_cols, "\n\n",
      "Numeric Columns: ", numeric_cols, "\n\n",
      "Numeric but Cetagoric Cols: ", categoric_but_numeric)
'''
Categoric Columns:  ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'] 

  Numeric Columns:  ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales'] 

  Numeric but Cetagoric Cols:  ['Outlet_Establishment_Year']

'''


# # Visualization and separation into quantiles(kuantillere ayrılması) of non-categorical values
# # Are There Numeric Columns's Outlier Value?
# # The control stage
# for col in numeric_cols: # Number Variable Analysis
#     print(dataUnderstand(df).num_summary(col, plot=True))
# for col in categoric_but_numeric: # NumButCat Variable Analysis
#     print(dataUnderstand(df).num_summary(col,plot=True))
# # Item_Outlet_Sales, Item_Visibility => There are Outlier Values
# print("********************************************************")
# df.loc[df["Item_Outlet_Sales"] > df["Item_Outlet_Sales"].quantile(0.95), "Item_Outlet_Sales"] = df["Item_Outlet_Sales"].quantile(0.95)
# df.loc[df["Item_Visibility"] > df["Item_Visibility"].quantile(0.95), "Item_Visibility"] = df["Item_Visibility"].quantile(0.95)
# # Outlier Values Dropped (Result Visualization)
# for col in numeric_cols:
#     print(dataUnderstand(df).num_summary(col, plot=True))

# # Correlation of Numerical Variables with each other
# dataUnderstand(df).corr_matrix(numeric_cols)


# # Target'a Gore Kolonlari Gruplama
# # Analysis of the relationship of numerical variables with Target
# for col in numeric_cols:
#     print(dataUnderstand(df).target_summary_with_num(col,target = "Item_Outlet_Sales"))
# # Analysis of the relationship of catButNum variables with Target
# for col in categoric_but_numeric:
#     print(dataUnderstand(df).target_summary_with_cat_or_catNum(col,target = "Item_Outlet_Sales"))
# # # Analysis of the relationship of categoric variables with Target
# for col in categoric_cols:
#     print(dataUnderstand(df).target_summary_with_cat_or_catNum(col,target = "Item_Outlet_Sales"))


# Are there missing values?
NaN_Columns, missing_Df = dataUnderstand(df).missingValueTables()
print("NaN Columns: ", NaN_Columns, "\n\n", missing_Df)

# Fill Missing Value
print(missingValue(df).categoric_Freq("Outlet_Size"),
      missingValue(df).median("Item_Weight"))
# Missing Values Test
NaN_Columns, missing_Df = dataUnderstand(df).missingValueTables()
print("NaN Columns: ", NaN_Columns, "\n\n", missing_Df)


# 2- DATA PREPROCESSING

# outlier value review of columns (sutunlarin aykiri deger incelenmesi)
for col in numeric_cols:
    print(col, ":", dataUnderstand(df).checkOutlier(col))

# capture of outliers as a single variable (Aykiri degerlerin tek degiskenli olarak yakalanmasi)
for col in numeric_cols:
    print(col, ":\n", dataUnderstand(
        df).catchOutliers(col, index=True, plot=True))

# crushing outliers (aykiri degerler baskilandi)
for col in numeric_cols:
    print(col, ":\n", dataPreProcess(df).crush_outliers(col, q1=0.25, q3=0.75))


# Outlier Values by very variable catch and deleted
result, df_scores = dataPreProcess(df).localOutlierFactor(numeric_cols, 20, 0.1, 50)
print("Result:\n", result, "Df_Scores:\n", df_scores)

df.drop(df[df_scores < -2.11425738].index,
        axis=0).reset_index(drop=True).shape[0]/len(df)
df = df.drop(df[df_scores < -2.11425738].index, axis=0).reset_index(drop=True)
# Control
result, df_scores = dataPreProcess(
    df).localOutlierFactor(numeric_cols, 20, 0.1, 50)
print("Result:\n", result, "Df_Scores:\n", df_scores)


# CATEGORIC COLUMNS => NUMERIC COLUMNS
#df = dataPreProcess(df).dummies(categoric_cols)

# for col in categoric_cols:
#     print(dataPreProcess(df).oneHotEncoder(col))
# for col in categoric_but_numeric:
#     print(dataPreProcess(df).labelEncoder(col))

#dataUnderstand(df).categoricVisualisationBarplot(categoric_cols, "Item_Outlet_Sales")  // BU KISMA BAKKKKKKKKKKKKKKK     
# plt.figure(figsize=(27,10))
# sns.barplot('Outlet_Identifier' ,'Item_Outlet_Sales', data=df ,palette='gist_rainbow')
# plt.xlabel('Outlet_Identifier', fontsize=14)
# plt.legend()
# plt.show()
# plt.figure(figsize=(10,5))
# sns.barplot('Outlet_Type' ,'Item_Outlet_Sales', data=df ,palette='nipy_spectral')
# plt.xlabel('Outlet_Type', fontsize=14)
# plt.legend()
# plt.show()
# plt.figure(figsize=(10,5))
# sns.barplot('Outlet_Size' ,'Item_Outlet_Sales', data=df ,palette='YlOrRd')
# plt.xlabel('Outlet_Size', fontsize=14)
# plt.legend()
# plt.show()
# plt.figure(figsize=(10,5))
# sns.barplot('Outlet_Size' ,'Item_Outlet_Sales', data=df ,palette='YlOrRd')
# plt.xlabel('Outlet_Size', fontsize=14)
# plt.legend()
# plt.show()

for col in categoric_cols:
    ohe = OneHotEncoder() 
    transformed = ohe.fit_transform(df[[col]])
    transformed_array = transformed.toarray()
    newDf = pd.DataFrame(transformed_array, columns=ohe.categories_)
    df = df.drop(col, axis=1)
    df = pd.concat([df, newDf], axis=1)
    print(df.shape)

for col in categoric_but_numeric:
    labelEncode = LabelEncoder()
    df[col] = labelEncode.fit_transform(df[col])
    print(df.shape)

from sklearn.utils.multiclass import type_of_target
# # 3- MODEL CREATE
# X_train,X_test,y_train, y_test = MachineLearning(df).get_dataset("Item_Outlet_Sales",0.3,12345)
# print(type_of_target(y_train))
# X_train, X_test = MachineLearning(df).standardScaler(X_train,X_test)
# y_train, y_test = MachineLearning(df).standardScaler(y_train.values.reshape(-1,1),y_test.values.reshape(-1,1))

# #y_train = labelEncode.fit_transform(y_train)
# #y_test = labelEncode.transform(y_test)

# cv_mean,acc_score = MachineLearning(df).modelCreate(cv=10,scoring="accuracy", target = "Item_Outlet_Sales",test_size=0.3,random_state=12345)
# print("Cv Mean: ",cv_mean,"\nAccuracy Score: ",acc_score)
# #   def modelCreate(self,cv,scoring,target,test_size,random_state):
# # .
# print(type_of_target(y_train))

print(df["Item_Outlet_Sales"].describe())



# columns to string
def col_to_str(columns):
    array = []
    for col in columns:
        col = ''.join(col)
        col = re.sub(" ","_",col)
        array.append(col)
    return array
df.columns = col_to_str(df.columns)
print(df.columns)

# target numeric to categoric
dataUnderstand(df).num_to_cat("Item_Outlet_Sales", 30, 500, 1500, 2500, 3200, 3700, 5000, 6600)


# 3- MODEL CREATE
X_train,X_test,y_train, y_test = MachineLearning(df).get_dataset(target = "Item_Outlet_Sales",
                                                                  test_size = 0.2,
                                                                  random_state = 42)

print(type_of_target(y_train)) # continuous => multiclass
X_train, X_test = MachineLearning(df).standardScaler(X_train,X_test)

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# rfc = RandomForestClassifier()
# rf_params = {"max_depth": [8, 15, None],
#              "max_features": [5, 7, "auto"],
#              "min_samples_split": [15, 20],
#              "n_estimators": [200, 300, 750, 1000]}
# gridSearchCV = GridSearchCV(rfc, rf_params, scoring="accuracy", n_jobs=-1, cv=10)
# gridSearchCV.fit(X_train, y_train) 
# bestParams = gridSearchCV.best_params_
# bestScore = gridSearchCV.best_score_ 
# print("Best Params: ",bestParams,"\nBest Score: ",bestScore)

# model,cv_mean, acc_score = MachineLearning(df).score("Item_Outlet_Sales",0.2,123)
# print("Model: {}\nCv Mean: {}\nAccuracy Score: {}\n".format(model,cv_mean,acc_score))


# FEATURE ENGINEERING
df["Weight/Low_Fat"] = df["Item_Weight"]/(df["Low_Fat"]+1)
df["Weight/Regular"] = df["Item_Weight"]/(df["Regular"]+1)
df["Weight/LF"] = df["Item_Weight"]/(df["LF"]+1)
df["Weight/reg"] = df["Item_Weight"]/(df["reg"]+1)
df["Weight/Type"] = df["Item_Weight"]/(df["Item_Type"]+1)
df["MRP/Weight"] = df["Item_MRP"]/df["Item_Weight"]
df["High/Weight"] = (df["High"]+1)/df["Item_Weight"]
df["Medium/Weight"] = (df["Medium"]+1)/df["Item_Weight"]
df["Small/Weight"] = (df["Small"]+1)/df["Item_Weight"]
df["Sales/Tier_1"] = df["Item_Outlet_Sales"]/(df["Tier_1"]+1)
df["Sales/Tier_2"] = df["Item_Outlet_Sales"]/(df["Tier_2"]+1)
df["Sales/Tier_3"] = df["Item_Outlet_Sales"]/(df["Tier_3"]+1)
df["Sales/Grocery_Store"] = df["Item_Outlet_Sales"]/(df["Grocery_Store"]+1)
df["Sales/Supermarket_Type1"] = df["Item_Outlet_Sales"]/(df["Supermarket_Type1"]+1)
df["Sales/Supermarket_Type2"] = df["Item_Outlet_Sales"]/(df["Supermarket_Type2"]+1)
df["Sales/Supermarket_Type3"] = df["Item_Outlet_Sales"]/(df["Supermarket_Type3"]+1)



# model,cv_mean, acc_score = MachineLearning(df).score("Item_Outlet_Sales",0.2,123)
# print("Model: {}\nCv Mean: {}\nAccuracy Score: {}\n".format(model,cv_mean,acc_score))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#log_reg = LogisticRegression() # %33
#rf = RandomForestClassifier() # %100
#gbc = GradientBoostingClassifier() # %100
#dtc = DecisionTreeClassifier() # %100
#xgb = XGBClassifier() # %100
# lgbm = LGBMClassifier() %100
test_summary, train_summary = MachineLearning(df).get_model(lgbm,"Item_Outlet_Sales",0.1,12345,"accuracy",10)
print("Test Result:\n ",test_summary,"\nTrain Result:\n",train_summary)



'''
# model, cv_mean, acc_score = MachineLearning(df).modelCreateRegressor("Item_Outlet_Sales",
#                                                              test_size=0.2,
#                                                              random_state=123,
#                                                              cv=10,scoring="accuracy")
'''

