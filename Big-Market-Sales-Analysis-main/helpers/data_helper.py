import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.dataPreProcessing import *
from sklearn.impute import SimpleImputer # for NaN Values
import random
    

class dataUnderstand(object):
     def __init__(self, df):
         self.df = df
     def __str__(self):
         sns.heatmap(self.df.corr(),annot=True, fmt=".2f",cmap="YlGn")
         plt.show()
         sns.pairplot(self.df)
         plt.show()
         return str(self.df.head())+"\n\n\n"+str(self.df.info())+"\n\n\n"+str(self.df.describe())+"\n\n\n"+str(self.df.isnull().sum())+"\n\n\n\n"
     def col_Corr(self,col): # numeric columns
         quantiles = [0.1,0.25,0.35,0.60,0.75,0.85,0.90,0.95]
         return self.df[col].describe(quantiles)
     def features(self):
         categoric_but_numeric = [col for col in self.df.columns if self.df[col].dtype == "O" and self.df[col].nunique() > 15]
         categoric_cols = [col for col in self.df.columns if self.df[col].dtype == "O" and self.df[col].nunique() <= 15 and 
                    col not in categoric_but_numeric]
         numeric_cols = [col for col in self.df.columns if not self.df[col].dtype == "O" and self.df[col].nunique() > 10]
         return categoric_cols,numeric_cols,categoric_but_numeric
     def target_summary_with_num(self,col,target = None):
         targetDf = pd.DataFrame({"Count":self.df[col].value_counts(),
                                    "Ratio":self.df[col].value_counts()/len(self.df),
                                    "Target_Mean":self.df.groupby(col)[target].mean()
                                    /len(self.df)}).sort_values("Count",ascending=False)
         
         return targetDf
     def target_summary_with_cat_or_catNum(self,col,target = None):
         targetDf = pd.DataFrame({"Count":self.df[col].value_counts(),
                                  "Ratio":self.df[col].value_counts()/len(self.df),
                                  "Target_Mean":self.df.groupby(col)[target].mean()
                                  /len(self.df)}).sort_values("Count",ascending=False)
         plt.figure(figsize=(15,5))
         sns.countplot(self.df[col],label="Count")
         plt.xticks(rotation=90)
         plt.legend()
         plt.show()
         return targetDf
     def num_summary(self,numeric_col, plot = False):
         if plot:
              self.df[numeric_col].hist(bins=20)
              plt.xlabel(numeric_col)
              plt.title(numeric_col)
              plt.show()
              sns.boxplot(self.df[numeric_col])
              plt.show()
         quantiles = [0.05, 0.2, 0.5, 0.7, 0.9, 0.95,0.99]
         return self.df[numeric_col].describe(quantiles).T 
     def corr_matrix(self,cols):
            fig = plt.gcf()
            fig.set_size_inches(15, 8)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            fig = sns.heatmap(self.df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                              cmap='GnBu')
            plt.show(block=True)
     def missingValueTables(self):
         NaN_Columns = [col for col in self.df.columns if self.df[col].isnull().sum() > 0]
         missingValues = self.df[NaN_Columns].isnull().sum()
         ratio = missingValues/len(self.df)
         typeCol = self.df[NaN_Columns].dtypes
         missingDf = pd.concat([missingValues,ratio,typeCol], keys=["missingValues","ratio","Col_Type"], axis=1)
         return NaN_Columns,missingDf
     
     def checkOutlier(self, col, q1 = 0.25, q3 = 0.75):
         lowerLimit,upperLimit = dataPreProcess(self.df).outlierThreshold(col,q1,q3)
         valuesDf = self.df[(self.df[col] < lowerLimit) | (self.df[col] > upperLimit)]
         if valuesDf.any(axis=None) == True:
             return True
         else:
             return False
     
     def catchOutliers(self,col, plot=False,q1=0.25, q3=0.75):
         if plot:
             sns.boxplot(self.df[col])
             plt.show()
         lower, upper = dataPreProcess(self.df).outlierThreshold(col,q1,q3)
         valuesDf = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
         if valuesDf.shape[0] > 10:
             return str(col) + "\n" + str(valuesDf.head()) + "\n\n" + str(valuesDf.index)
         else:
             return str(col) + "\n" + str(valuesDf) + "\n\n" + str(valuesDf.index)
     def categoricVisualisationBarplot(self,categoric_cols,target):
         number_of_colors = len(categoric_cols)
         color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(number_of_colors)]
         COLORS = [(139, 0, 0), 
               (0, 100, 0),
               (0, 0, 139)]
         if type(categoric_cols) == list:
             for i in range(len(categoric_cols)):
                 plt.figure(figsize=(27,10))
                 sns.barplot(categoric_cols[i], target, data=self.df, palette="gist_rainbow")
                 #sns.barplot(categoric_cols[i] ,target, data=self.df ,palette=random.choice(COLORS))
                 plt.xlabel('Item_Type', fontsize=14)
                 plt.legend()
                 plt.xticks(rotation=60,fontsize=14)
                 plt.title("{} of {}".format(categoric_cols[i],target))
                 plt.show()
         else:
                 plt.figure(figsize=(27,10))
                 sns.barplot(categoric_cols, target, data=self.df, palette="gist_rainbow")
                 plt.xlabel('Item_Type', fontsize=14)
                 plt.legend()
                 plt.xticks(rotation=60,fontsize=14)
                 plt.title("{} of {}".format(categoric_cols,target))
                 plt.show()
     def categoricVisualisationCountplot(self,categoric_cols):
         if type(categoric_cols) == list:
             for col in categoric_cols:
                 plt.figure(figsize=(25,10))
                 sns.countplot(data = self.df, x=col)
                 plt.xticks(rotation=60, fontsize=15)
                 plt.title("{} Count".format(col))
                 plt.show()
         else:
            plt.figure(figsize=(25,10))
            sns.countplot(data = self.df, x=categoric_cols)
            plt.xticks(rotation=60, fontsize=15)
            plt.title("{} Count".format(categoric_cols))
            plt.show()
     def categoricVisualisationBoxplot(self,categoric_cols,target):
          if type(categoric_cols) == list:
              for col in categoric_cols:
                  plt.figure(figsize=(25,10))
                  sns.boxplot(data = self.df, x = col, y = target)
                  plt.xticks(rotation=60, fontsize=15)
                  plt.title("{} of {}".format(col,target))
                  plt.show()
          else:
                plt.figure(figsize=(25,10))
                sns.boxplot(data = self.df, x = categoric_cols, y = target)
                plt.xticks(rotation=60, fontsize=15)
                plt.title("{} of {}".format(categoric_cols,target))
                plt.show()
     def categoricVisualisationStripplot(self,categoric_cols,target):
         if type(categoric_cols) == list:
             for col in categoric_cols:
                 plt.figure(figsize=(25,10))
                 sns.stripplot(data = self.df, x = col, y = target)
                 plt.xticks(rotation=60, fontsize=15)
                 plt.title("{} of {}".format(col,target))
                 plt.show()  
         else:
                 plt.figure(figsize=(25,10))
                 sns.stripplot(data = self.df, x = categoric_cols, y = target)
                 plt.xticks(rotation=60, fontsize=15)
                 plt.title("{} of {}".format(categoric_cols,target))
                 plt.show() 
     def categoricVisualisationViolinplot(self,categoric_cols,target):
         if type(categoric_cols) == list:
             for col in categoric_cols:
                 plt.figure(figsize=(25,10))
                 sns.stripplot(data = self.df, x = col, y = target)
                 plt.xticks(rotation=60, fontsize=15)
                 plt.title("{} of {}".format(col,target))
                 plt.show()
         else:
                 plt.figure(figsize=(25,10))
                 sns.stripplot(data = self.df, x = categoric_cols, y = target)
                 plt.xticks(rotation=60, fontsize=15)
                 plt.title("{} of {}".format(categoric_cols,target))
                 plt.show()
     def numericVisulisation(self,numeric_cols):
         for col in numeric_cols:
             sns.distplot(self.df[col])
             plt.title("Graph of density by {}".format(col))
             plt.show()          
     def num_to_cat(self,feature,a,b,c,d,e,f,g,h):
         self.df.loc[(self.df[feature] > a) & (self.df[feature] <= b), feature] = 0
         self.df.loc[(self.df[feature] > b) & (self.df[feature] <= c), feature] = 1
         self.df.loc[(self.df[feature] > c) & (self.df[feature] <= d), feature] = 2
         self.df.loc[(self.df[feature] > d) & (self.df[feature] <= e), feature] = 3
         self.df.loc[(self.df[feature] > e) & (self.df[feature] <= f), feature] = 4
         self.df.loc[(self.df[feature] > f) & (self.df[feature] <= g), feature] = 5
         self.df.loc[(self.df[feature] > g) & (self.df[feature] <= h), feature] = 6
    
class missingValue(object): # Machine Learning Algorithm da kullabılabilir.
    def __init__(self,df):
        self.df = df
    def sıfır(self,missing_col):
        self.df[missing_col] = self.df[missing_col].fillna(0)
        #self.df[missing_col] = self.df[col].replace(to_replace=0, value=np.NaN)
        return self.missingVisualization(missing_col,plot=True)
    def average(self,missing_col):
        imputer = SimpleImputer(missing_values = "NaN", strategy = "mean")
        imputer_col = imputer.fit(self.df[missing_col].values)
        self.df[col] = imputer_col.transform(self.df[col])
        return self.missingVisualization(missing_col)
        # self.df[missing_col] = self.df[missing_col].fillna(self.df[missing_col].mean())
        # return self.missingVisualization(missing_col)
    def median(self, missing_col):
        self.df[missing_col] = self.df[missing_col].fillna(self.df[missing_col].median())
        return self.missingVisualization(missing_col)
    def mode(self,missing_col):
        self.df[missing_col] = self.df[missing_col].fillna(self.df[missing_col].mode())
        return self.missingVisualization(missing_col)
    def linear(self,missing_col):
        self.df[missing_col] = self.df[missing_col].interpolate(method = "linear")
        return self.missingVisualization(missing_col)
    def standard_deviation(self,missing_col):
        self.df[missing_col].fillna(self.df[missing_col].std(), inplace=True)
        return self.missingVisualization(missing_col)
    def categoric_Freq(self,missing_col):
        self.df[missing_col].fillna(self.df[missing_col].describe().top, inplace=True)
        return self.missingVisualization(missing_col)
    def missingVisualization(self,missing_col, plot=False):
        if plot:
            self.df[missing_col].hist(bins=20, figsize=(10,8))
            plt.xlabel(missing_col)
            plt.title(missing_col)
            plt.show()
        return  missing_col +" missing value: "+str(self.df[missing_col].isnull().sum())
    def corr_columns(self, col1, col2):
        return self.df[[col1,col2]].corr()
