import pandas as pd

import numpy as np
from helpers.data_helper import *
from helpers.machineLearning import *

# DATA PREPROCESSING
df = pd.read_csv("datas/TotalDatas/totalData.csv")
# print(dataUnderstand(df))
# for col in df.columns:
#     print(dataUnderstand(df).col_Corr(col))
    
categoric_cols,numeric_cols,categoric_but_numeric = dataUnderstand(df).features()
# for col in numeric_cols:
#     targetDf = dataUnderstand(df).target_summary_with_num(col,target = "NumberEntryToSite")
#     print(targetDf)
    
# for col in categoric_cols:
#     print(dataUnderstand(df).target_summary_with_Cat(col,"NumberEntryToSite"))

    
# for col in numeric_cols:
#     print(dataUnderstand(df).num_summary(col,plot=True))

def outlierValues(df,col):
    q1 = df[col].quantile(0.1)
    q3 = df[col].quantile(0.9)
    iqr = q3 - q1 
    lowerLimit = q1 - iqr * 1.5 # -4.5
    upperLimit = q3 + iqr * 1.5 # 15.5
    return df.loc[(df["RepeatNumber"] < lowerLimit) | (df["RepeatNumber"] > upperLimit)]
# print(outlierValues(df, "NumberEntryToSite"),"\n\n",outlierValues(df, "RepeatNumber"))
# print(dataUnderstand(df).catchOutliers("RepeatNumber",plot=True,q1 = 0.1, q3 = 0.9)) # 5 rows

'''
     Minute  Second  RepeatNumber  ...        sport      dport NumberEntryToSite
325      12      19            30  ...  sport=61820   dport=53                 2
326      12      19            30  ...  sport=53317  dport=443                 1
327      12      19            30  ...  sport=53322  dport=443                 1
328      12      19            30  ...  sport=53323  dport=443                 1
582      23      12            38  ...  sport=53449  dport=445                 1
'''
# print(dataUnderstand(df).catchOutliers("NumberEntryToSite",plot=True, q1 = 0.1, q3 = 0.9)) # 5 rows
'''
NumberEntryToSite
       Minute  Second  ...  dport NumberEntryToSite
16057       2      30  ...      0               696
16058       2      30  ...      0               696
16059       2      30  ...      0               696
16060       2      30  ...      0               696
16061       2      30  ...      0               696
'''
df.loc[df["RepeatNumber"] > df["RepeatNumber"].quantile(0.95),"RepeatNumber"] = df["RepeatNumber"].quantile(0.95)
df.loc[df["NumberEntryToSite"] > df["NumberEntryToSite"].quantile(0.95), "NumberEntryToSite"] = df["RepeatNumber"].quantile(0.95)
# for col in numeric_cols: # outlier value'den sonra control
#     print(dataUnderstand(df).num_summary(col,plot=True))

df = pd.get_dummies(data = df, columns=categoric_cols, drop_first=True)
for col in categoric_but_numeric:
    print(dataPreProcess(df).labelEncoder(col))



# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier 
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,AdaBoostRegressor,GradientBoostingRegressor
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn.svm import SVC # Support Vector Classifier
# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score, classification_report


# from sklearn.metrics import confusion_matrix,accuracy_score
# X = df.drop("NumberEntryToSite",axis=1)
# y = df["NumberEntryToSite"]
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
'''
## Grid Search CV
# xgboost = XGBClassifier()
# parameters = {
#     'max_depth': range (2, 10),
#     'n_estimators': range(20),
#     'learning_rate': [0.1, 0.01, 0.05]
# }

# grid_search = GridSearchCV(
#     estimator = xgboost, # xgb, kmeans, knn, mlp
#     param_grid=parameters,
#     scoring = 'accuracy',
#     n_jobs = 10,
#     cv = 10,
#     verbose=True
# )
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print("Best params: ",best_params,"\nBest Score: ",best_score)
'''

# xgboost = XGBClassifier()
# # knn = KNeighborsClassifier()
# # mlp = MLPClassifier()
# # kmeans = KMeans()
# # rfc = RandomForestClassifier()
# # abc = AdaBoostClassifier()
# # svc = SVC()
# # lgbm = LGBMClassifier()
# # catboost = CatBoostClassifier()  
# xgboost.fit(X_train,y_train)
# y_pred = xgboost.predict(X_test)
# X_pred = xgboost.predict(X_train)
# cv_result = cross_val_score(xgboost, X_train, y_train, cv=10, n_jobs = -1, scoring="accuracy")
# cv_mean = cv_result.mean()
# plt.figure(figsize=(20,10))
# sns.heatmap(confusion_matrix(y_test,y_pred))
# plt.show()
# print("CV Mean: ",cv_mean,
#       "\nAccuracy Score: ",accuracy_score(y_test, y_pred),
#       "\nMean Squared Error: ",mean_squared_error(y_test, y_pred),
#       "\nMean Absolute Error: ",mean_absolute_error(y_test, y_pred),
#       "\nR^2 Score: ",r2_score(y_test,y_pred),
#       "\nClassification Report Test:\n ",classification_report(y_test, y_pred))
# print("Classification Report Train:\n",classification_report(y_train, X_pred))



'''
XGBClassifier               

        CV Mean:  0.8343018466237794 
        Accuracy Score:  0.8311579651941098 
        Mean Squared Error:  0.6651606425702812 
        Mean Absolute Error:  0.22305890227576974 
        R^2 Score:  0.9944022082590656
        Classification Report Test:
                        precision    recall  f1-score   support

                   1       0.83      0.94      0.88      3559
                   2       0.64      0.41      0.50      1076
                   3       0.74      0.42      0.53       177
                   4       0.82      0.77      0.79        81
                   5       0.69      0.79      0.74        43
                   6       0.81      0.50      0.62        26
                   7       0.83      0.95      0.89        21
                   8       0.80      0.71      0.75        17
                   9       1.00      0.82      0.90        28
                  10       0.95      1.00      0.97        18
                  11       1.00      0.67      0.80         6
                  12       1.00      1.00      1.00        33
                  13       1.00      1.00      1.00         3
                  14       1.00      1.00      1.00        16
                  15       1.00      1.00      1.00       297
                  16       1.00      1.00      1.00        10
                  18       1.00      1.00      1.00        15
                  20       1.00      1.00      1.00        49
                  21       0.84      1.00      0.91        26
                  22       1.00      0.90      0.95        30
                  23       1.00      1.00      1.00         9
                  24       1.00      1.00      1.00        12
                  25       1.00      0.90      0.95        10
                  26       1.00      1.00      1.00        13
                  28       1.00      1.00      1.00        22
                  29       1.00      1.00      1.00         7
                  31       1.00      1.00      1.00         8
                  32       1.00      1.00      1.00        16
                  33       1.00      1.00      1.00        10
                  34       1.00      1.00      1.00         8
                  36       0.98      1.00      0.99        56
                  37       1.00      0.97      0.99        37
                  39       1.00      1.00      1.00        14
                  40       1.00      1.00      1.00        49
                  42       1.00      1.00      1.00        10
                  43       1.00      1.00      1.00         9
                  44       0.96      1.00      0.98        23
                  45       1.00      1.00      1.00        18
                  46       1.00      1.00      1.00        21
                  49       1.00      0.94      0.97        17
                  52       1.00      1.00      1.00        12
                  53       1.00      1.00      1.00        14
                  54       1.00      1.00      1.00        17
                  56       1.00      1.00      1.00        14
                  58       1.00      1.00      1.00        19

            accuracy                           0.83      5976
           macro avg       0.95      0.93      0.94      5976
        weighted avg       0.82      0.83      0.82      5976

        Classification Report Train:
                       precision    recall  f1-score   support

                   1       0.91      0.99      0.95      8348
                   2       0.93      0.69      0.80      2406
                   3       0.99      0.78      0.87       375
                   4       1.00      1.00      1.00       207
                   5       1.00      1.00      1.00       157
                   6       1.00      1.00      1.00        46
                   7       1.00      1.00      1.00        56
                   8       1.00      1.00      1.00        39
                   9       1.00      1.00      1.00        44
                  10       1.00      1.00      1.00        42
                  11       1.00      1.00      1.00         5
                  12       1.00      1.00      1.00        63
                  13       1.00      1.00      1.00        23
                  14       1.00      1.00      1.00        40
                  15       1.00      1.00      1.00       729
                  16       1.00      1.00      1.00        22
                  18       1.00      1.00      1.00        39
                  20       1.00      1.00      1.00        91
                  21       1.00      1.00      1.00        79
                  22       1.00      1.00      1.00        80
                  23       1.00      1.00      1.00        14
                  24       1.00      1.00      1.00        12
                  25       1.00      1.00      1.00        15
                  26       1.00      1.00      1.00        13
                  28       1.00      1.00      1.00        62
                  29       1.00      1.00      1.00        22
                  31       1.00      1.00      1.00        23
                  32       1.00      1.00      1.00        48
                  33       1.00      1.00      1.00        23
                  34       1.00      1.00      1.00        26
                  36       1.00      1.00      1.00       124
                  37       1.00      1.00      1.00        74
                  39       1.00      1.00      1.00        25
                  40       1.00      1.00      1.00       111
                  42       1.00      1.00      1.00        32
                  43       1.00      1.00      1.00        34
                  44       1.00      1.00      1.00        65
                  45       1.00      1.00      1.00        27
                  46       1.00      1.00      1.00        71
                  49       1.00      1.00      1.00        32
                  52       1.00      1.00      1.00        40
                  53       1.00      1.00      1.00        39
                  54       1.00      1.00      1.00        37
                  56       1.00      1.00      1.00        42
                  58       1.00      1.00      1.00        39

            accuracy                           0.93     13941
           macro avg       1.00      0.99      0.99     13941
        weighted avg       0.93      0.93      0.93     13941
        
Precision => Pozitif olarak tahminledigimiz degerlerin yuzde kaci pozitif
Recall => Pozitif olarak tahmin etmemiz gereken islemlerin yuzde kaci pozitif


KNeighborsClassifier         => 0.6845716198125836
MLPClassifier                => 0.7071619812583668
KMeans                       => 0.2362784471218206
RandomForestClassifier       => 0.8017068273092369
AdaBoostClassifier           => 0.631024096385542
Support Vector Classifier    => 0.6701807228915663
LGBMClassifier               => 0.5582329317269076 
CatBoostClassifier           => 0.802376171352075 
'''










