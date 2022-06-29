import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor   # lof doesn't accept missing values encoded as NaN
from sklearn.ensemble import HistGradientBoostingClassifier # hgb accept missing values encoded as NaN
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class dataPreProcess:
    def __init__(self,df):
        self.df = df
    def outlierThreshold(self,col,q1 = 0.01, q3 = 0.99):
        Q1 = self.df[col].quantile(q1)
        Q3 = self.df[col].quantile(q3)
        iqr = Q3 - Q1
        lowerLimit = Q1 - iqr * 1.5
        upperLimit = Q3 + iqr * 1.5
        return lowerLimit,upperLimit
    def crush_outliers(self,col,q1 = 0.01,q3 = 0.99):
        lowerLimit,upperLimit = dataPreProcess(self.df).outlierThreshold(col,q1,q3)
        self.df.loc[(self.df[col] < lowerLimit), col] = lowerLimit
        self.df.loc[(self.df[col] > upperLimit), col] = upperLimit       
    def localOutlierFactor(self, col, neighbors = None, percent = None, plot_xlim = None): # Unsupervised Algorithm
        # 1: Normal, -1: Anormal    
        lof = LocalOutlierFactor(n_neighbors = neighbors, contamination=percent)
        lof.fit_predict(self.df[col])
        df_scores = lof.negative_outlier_factor_
        plt.style.use("bmh")
        pd.DataFrame(np.sort(df_scores)).plot(stacked = True, xlim = [0,plot_xlim], style = ".-", figsize = (10,5))
        plt.show()
        result = np.sort(df_scores)[0:plot_xlim]
        return result,df_scores
    def dummies(self,categoric_cols):
        self.df = pd.get_dummies(self.df, columns = categoric_cols, drop_first=True), # Male:1,Female(default):0
        return self.df 
    def oneHotEncoder(self,categoric_col):
        ohe = OneHotEncoder()
        transformed = ohe.fit_transform(self.df[[categoric_col]])
        transformed_array = transformed.toarray()
        newDf = pd.DataFrame(transformed_array, columns=ohe.categories_)
        self.df = self.df.drop(categoric_col,axis=1)
        self.df = pd.concat([self.df,newDf], axis=1)
        return self.df.shape
    def labelEncoder(self,categoric_but_numeric):
        self.df[categoric_but_numeric] = LabelEncoder().fit_transform(self.df[categoric_but_numeric])
        return self.df[categoric_but_numeric].shape

'''

for col in categoric_cols:
    print(dataPreProcess(df).oneHotEncoder(col))
for col in categoric_but_numeric:
    print(dataPreProcess(df).labelEncoder(col))

(8503, 16)
(8503, 21) -4
(8503, 14) -13
(8503, 14) -15
(8503, 15) -17
(8503,)  -32
(8503,) -32

labelEncode = LabelEncoder()
for col in categoric_cols:
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[[col]])
    transformed_array = transformed.toarray()
    newDf = pd.DataFrame(transformed_array, columns=ohe.categories_)
    df = df.drop(col, axis=1)
    df = pd.concat([df, newDf], axis=1)


for col in categoric_but_numeric:
    df[col] = labelEncode.fit_transform(df[col])
'''


