from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,Normalizer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.data_helper import *
from helpers.dataPreProcessing import *
from helpers.machineLearning import *


columns = ["Month", "Day", "Time", "Server", "Unnamed", "ID", "TR_IST_AP", "flow_or_url", "allow_or_src", "SNAT_or_DNAT", "mac_or_dst",
           "mac_or_request", "protocol", "sport", "dport"]
df = pd.read_csv("datas/turcom-wifi.txt", error_bad_lines=False,
                 sep=" ", header=None, nrows=10000, names=columns)

df = df.drop("Unnamed", axis=1)
# df.to_csv("datas/processedDatas/processFile.csv")


def datetime(df, time):
    df[time] = pd.to_datetime(df[time])
    df["Minute"] = df[time].dt.minute
    df["Second"] = df[time].dt.second


datetime(df, "Time")
df = df.drop(["Time","ID"],axis=1)
print(dataUnderstand(df))
df_copy = df.copy()

# NaN_Columns,missingDf = dataUnderstand(df_copy).missingValueTables()
# print("NaN_Columns: ",NaN_Columns,"\n ",missingDf)
# df.fillna(0,axis=1, inplace=True)
# df_copy["Month"] = 6
# df_copy["Date"] = df_copy.agg(lambda x: f"2022-{x['Month']}-{x['Day']} 00:{x['Minute']}:{x['Second']}", axis=1)
# df_copy["Date"] = pd.to_datetime(df_copy["Date"])
# print(type(df_copy["Date"][0]))
# df_copy = df_copy.drop(["Month","Day","Minute","Second"], axis=1)
# df_copy.to_csv("datas/processedDatas/mongo-wifi.csv", index = False)
# # for value in df_copy["Date"]:
# #     print(value)


# categoric_cols,numeric_cols,categoric_but_numeric = dataUnderstand(df).features()
# NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
# print("NaN_Columns: ",NaN_Columns,"\n ",missingDf)

# for col in NaN_Columns:
#     missingValue(df).categoric_Freq(col)
    
# NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
# print("NaN_Columns: ",NaN_Columns,"\n ",missingDf)

# for col in df.columns:
#     print(dataUnderstand(df).col_Corr(col))

# # # Outlier Value Control
# # for col in df.columns:
# #     print(dataUnderstand(df).checkOutlier(col))

# # for col in numeric_cols: # target'a gore numeric col oranı
# #     print(col,"\n",dataUnderstand(df).target_summary_with_num(col,"Second"))

# for col in categoric_cols: # target'a gore numeric col oranı and visualization
#     print(dataUnderstand(df).target_summary_with_cat_or_catNum(col,"Second"))

# for col in numeric_cols: # numeric col outlier value control
#     print(dataUnderstand(df).num_summary(col,plot=True))

# def dummies(dataFrame, categoric_cols):
#     dataFrame = pd.get_dummies(dataFrame,columns=categoric_cols,drop_first=True)
#     return dataFrame
# df = dummies(df, categoric_cols) 

# # Bu yontem ile columns yapilari bozuluyor
# # for col in categoric_cols:
# #     ohe = OneHotEncoder()
# #     transformed = ohe.fit_transform(df[[col]])
# #     transformed_array = transformed.toarray()
# #     newDf = pd.DataFrame(data=transformed_array,columns=ohe.categories_)
# #     df = df.drop(col,axis=1)
# #     df = pd.concat([df,newDf],axis=1)
# #     print(df.shape)

# def labelEncoder(df,categoric_but_numeric):
#     le = LabelEncoder()
#     df[categoric_but_numeric] = le.fit_transform(df[categoric_but_numeric])
#     return df.shape

# for col in categoric_but_numeric:
#     print(labelEncoder(df, col))

# df = df.drop(["protocol_UNKNOWN","protocol_]","protocol_request:"], axis=1)
# print(df.columns)
# X_train,X_test,y_train, y_test = MachineLearning(df).get_dataset("flow_or_url_flows",0.3,123)
# X_train, X_test = MachineLearning(df).standardScaler(X_train, X_test)

# # from sklearn.model_selection import cross_val_score
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# # lr = LogisticRegression()
# # lr.fit(X_train, y_train)
# # y_pred = lr.predict(X_test)
# # cv_result = cross_val_score(lr, X_train,y_train, cv=10,scoring="accuracy")
# # cv_mean = cv_result.mean()
# # acc_score = accuracy_score(y_test,y_pred)
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt = ".2f", cmap="Greens")
# # plt.title("Log Reg: {}".format(acc_score))
# # plt.show()
# # print("CV Mean: ",cv_mean,"\nAcc Score:",acc_score)


# # resultDf = MachineLearning(df).score("flow_or_url_flows",0.3,123)
# # print(resultDf)
# '''
#                         cv_mean  acc_score
# LogisticReg            0.999857   1.000000
# KNN                    0.999714   1.000000
# SupportVectorMachines  0.999571   0.999000
# DecisionTree           0.999857   1.000000
# RandomForest           0.999857   1.000000
# Adaboost               0.999857   1.000000
# GradientBoost          0.999857   1.000000
# XGBoost                0.999857   0.999667
# DecisionTree           0.999857   1.000000
# LightGBM               0.999714   0.999667
# CatBoost               0.999857   1.000000

# '''
# # from sklearn.linear_model import LogisticRegression
# # lr = LogisticRegression()
# # pre_score,fScore, test_summary, train_summary = MachineLearning(df).get_model(lr,"flow_or_url_flows",0.2,123,"accuracy",10)
# # print("Pre Score: ",pre_score,"\nF-Score:",fScore,"\nTrain Result:\n",train_summary,"\nTest Result:\n",test_summary)

# '''
# Pre Score:  1.0 
# F-Score: 1.0 
# Train Result:
#                 precision    recall  f1-score   support

#             0       1.00      1.00      1.00      1708
#             1       1.00      1.00      1.00      6292

#     accuracy                           1.00      8000
#     macro avg       1.00      1.00      1.00      8000
# weighted avg       1.00      1.00      1.00      8000
 
# Test Result:
#                 precision    recall  f1-score   support

#             0       1.00      1.00      1.00       416
#             1       1.00      1.00      1.00      1584

#     accuracy                           1.00      2000
#     macro avg       1.00      1.00      1.00      2000
# weighted avg       1.00      1.00      1.00      2000
'''

'''



