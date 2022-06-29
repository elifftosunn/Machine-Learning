import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from helpers.data_helper import *
from helpers.dataPreProcessing import *
from helpers.machineLearning import *

# sport => Source Port => hedef baglanti noktasini belirtir(wifi)
# dport => Destination Port => disaridan wifiye gelen port 
# dst => Destination Support Services
# src => Source (Kablosuz ag) 
# dst => Destination (hedef)  => disaridan gelen ip, dst_port and protocol(tcp,udp gibii)
# udp => type of packet, simpler packets with just a destination, data inself and a checksum
# tcp => type of packet, include extra protocol information
# flows => herhangi internet baglantisina ihtiyac duymadan yalnizca bluetooth veya P2P Wi-Fi ile uygulamaya giris yaparken diger kullanicilar ile mesajlasabiliyorsunuz
# mac => wifi'si olmayan bir agda internet erisiminiz varsa wifisini paylasarak bir mac pc'yi kendi gecici yonlendiriciniz olarak kullanabilirsiniz

columns = ["Month","Day","Time","Server","Unnamed","ID","TR_IST_AP","flow_or_url","allow_or_src","SNAT_or_DNAT","mac_or_dst",
           "mac_or_request","protocol","sport","dport"]
df = pd.read_csv("datas/turcom-wifi.txt",error_bad_lines=False,
                 sep = " ",header=None, nrows=10000,names = columns)

df = df.drop("Unnamed",axis=1)
# df.to_csv("datas/processedDatas/processFile.csv")
def datetime(df,time):
    df[time] = pd.to_datetime(df[time])
    df["Minute"] = df[time].dt.minute
    df["Second"] = df[time].dt.second
datetime(df, "Time")    
print(dataUnderstand(df))    
df_copy = df.copy()
 
# FEATURE ENGINEERING   
# - kac tane disaridan dst geliyor ve kac saniyede geliyor
def separatingColon(df,col,split,sep1,sep2):
    firstValues, secondValues = [],[]
    firstCount, secondCount = 0,0
    for col in df[col].values:
        col = str(col)
        values = col.split(split)
        if values[0] == sep1:
            firstValues.append(values[1])
            firstCount += 1
        if values[0] == sep2:
            secondValues.append(values[1])
            secondCount += 1
    return firstCount,secondCount,firstValues,secondValues

dstCount,macCount,dstValues,macValues = separatingColon(df, "mac_or_dst", "=", "dst", "mac")
print("dst count: ",dstCount,"\nmac count: ",macCount)       
print(df["mac_or_dst"].describe())
dstDf = pd.DataFrame(data=dstValues,columns=["DST"])
macDf = pd.DataFrame(data=macValues, columns = ["MAC"])
df = pd.concat([df,dstDf,macDf],axis=1)
df["DST"] = df["DST"].fillna(0)
df["MAC"] = df["MAC"].fillna(0)
print(df["DST"].describe())
print(df["MAC"].describe())
# Categoric Columns Visualisation
dataUnderstand(df).categoricVisualisationBarplot("MAC","Second")
dataUnderstand(df).categoricVisualisationCountplot("MAC")
dataUnderstand(df).categoricVisualisationBoxplot("MAC","Second") # outlier values convert to number
# dataUnderstand(df).categoricVisualisationBoxplot("DST","Second") # outlier values convert to number(nunique value very much)

df.to_csv("datas/processedDatas/data.csv")
# - hangi protocol ile geliyor ve protocolun gelme sureleri nedir(tcp-udp ayrı ayrı)
print(df.protocol.value_counts())
print(df.protocol.values)
# udpSecondTotal, tcpSecondTotal = 0,0
# for col in df.columns:
#     if col == "protocol":          
#         for value in df[col].values:      
#             if value == "protocol=udp":
#                 print(df["Minute"].unique())
#                     #udpSecondTotal += s
# print("udpSecondTotal: ",udpSecondTotal)
#                 #udpSecondTotal += df["Second"].values
#                 #print(df.iloc[value,"Second"])
#             # if value == "protocol=tcp":
#             #     tcpSecondTotal += df["Second"].values
# #          self.df.loc[(self.df[feature] > a) & (self.df[feature] <= b), feature] = 0


# print(df.columns)
# totalValues = [] 
# values = {}
# count,minute, second = 0, 2, 26 
# for i in range(len(df)): # her satırda dolas 
#     for j in range(len(df.columns)): # her sutunda dolas
#         if df.iloc[i,j] == "protocol=udp":
#             for arr in df.loc[i,["Minute","Second"]]: # 0.row Minute,Second columns, 1.row ....., 2.row
#                 arr = np.array(df.loc[i,["Minute","Second"]], dtype=np.int64) # 0.row minute, 1.row minute
#                 a,b = arr             
#                 if a == minute and b == second: # 1.row == 2.minute, 2.row == 2.minute, 3.row 2.minute
#                     print(df.loc[i,["Minute","Second","protocol"]])
#                     count += 1
#                     minute += 1
#                     second += 1
# values["count"] = count
# totalValues.append(values)
# print("totalValues: ",totalValues)            


'''

Second                26
protocol    protocol=udp
Name: 0, dtype: object
Values:  {'Minute': 2, 'Second': 26, 'Count': 1}
Minute                 2
Second                26
protocol    protocol=udp
Name: 1, dtype: object
Values:  {'Minute': 2, 'Second': 26, 'Count': 1}
Minute                 2
Second                26
protocol    protocol=udp
Name: 1, dtype: object
Values:  {'Minute': 2, 'Second': 26, 'Count': 1}
Minute                 2
Second                26
protocol    protocol=udp
Name: 2, dtype: object
Values:  {'Minute': 2, 'Second': 26, 'Count': 1}
Minute                 2
Second                26
protocol    protocol=udp
Name: 2, dtype: object

Month                                         Jun
Day                                            27
Time                          2022-06-29 00:43:00
Server            server-176.53.2.142.as42926.net
ID                           1656279780.353035768
TR_IST_AP                             TR_IST_AP_4
flow_or_url                                 flows
allow_or_src                                allow
SNAT_or_DNAT                     src=172.19.0.173
mac_or_dst                      dst=172.18.14.225
mac_or_request              mac=F4:46:37:8A:F5:A7
protocol                             protocol=udp
sport                                 sport=53535
dport                                    dport=53
Minute                                         43
Second                                          0
DST                                             0
MAC                                             0
Name: 9996, dtype: object

'''
    #print(df.iloc[2,0]) # row,column
#print(df.loc[["protocol","Minute"]])
 # self.df.loc[(self.df[feature] > a) & (self.df[feature] <= b), feature] = 0

    #print(df.loc[i,"protocol"])
# for i in df:
#     print(df.loc[i,"protocol"].values)     
# print("UDP Second Total: ",udpSecondTotal,"\nTCP Second Total: ",tcpSecondTotal)





# df["UDP"] = df["UDP"].fillna(0)
# df["TCP"] = df["TCP"].fillna(0)
# dataUnderstand(df).categoricVisualisationBarplot("TCP","Second")
# dataUnderstand(df).categoricVisualisationCountplot("TCP")
# dataUnderstand(df).categoricVisualisationBoxplot("TCP","Second")
# sns.set_theme(style="dark")
# g = sns.regplot(data=df,x="Second",y="")




# datetime(df, "Time")
# df = df.drop("Time",axis=1)
    
# for col in df.columns:
#     print(col,":\n",df[col].value_counts())

# categoric_cols,numeric_cols,categoric_but_numeric = dataUnderstand(df.drop("ID",axis=1)).features()
# print("categoric_cols: ",categoric_cols,"\nnumeric_cols",numeric_cols,"\ncategoric_but_numeric: ",categoric_but_numeric)


# # Numeric Columns Visualisation
# dataUnderstand(df).numericVisulisation(numeric_cols)



# for col in df.columns:
#     print(dataUnderstand(df).col_Corr(col))
# '''
# Maximum flows yani herhangi internet baglantisina ihtiyac duymadan yalnizca bluetooth veya P2P Wi-Fi ile uygulamaya giris yaparken diger kullanicilar ile mesajlasabiliyorsunuz
# Maximum src yani SNAT: "ic" agdaki birden cok ana pc'nin "dis" agdaki herhangi bir ana pc'ye erismesine izin verir.
# Maximum TR_IST_AP_4
# Maximum protocol=udp yani veri gonderir ve alır, hızlıdır ama veri gonderimini garanti etmez
# Maximum sport: https://api.opendns.com/... yani source port
# Maximum dport: 53 yani hedef(destination) port 

# '''



# # Categoric Columns Visualisation
# # dataUnderstand(df).categoricVisualisationBarplot(categoric_cols,"Second")
# # dataUnderstand(df).categoricVisualisationCountplot(categoric_cols)
# # dataUnderstand(df).categoricVisualisationBoxplot(categoric_cols,"Second")
# # dataUnderstand(df).categoricVisualisationStripplot(categoric_cols,"Second")
# # dataUnderstand(df).categoricVisualisationViolinplot(categoric_cols,"Second")
# # dataUnderstand(df).categoricVisualisationCountplot(categoric_but_numeric)


# # Missing Value Control and Fillna
# NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
# print("NaN_Columns: ",NaN_Columns,"\n",missingDf)
# # NaN_Columns:  ['protocol', 'sport', 'dport'] 
# for col in NaN_Columns:
#     missingValue(df).categoric_Freq(col)
# NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
# print("NaN_Columns: ",NaN_Columns,"\n",missingDf)


# # categoric cols => numeric col
# df = df.drop("ID",axis=1)
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# for col in categoric_cols:
#     ohe = OneHotEncoder()
#     transformed = ohe.fit_transform(df[[col]])
#     transformed_array = transformed.toarray()
#     newDf = pd.DataFrame(transformed_array, columns=ohe.categories_)
#     df = df.drop(col,axis=1)
#     df = pd.concat([df,newDf], axis=1)
#     print(df.shape)
# for col in categoric_but_numeric:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     print(df.shape)

# print(df.columns)
# # correcting the structure of the column
# import re
# def columnsToString(df,columns):
#     array = []
#     for col in df.columns:
#         col = "".join(col)
#         col = re.sub("/","_",col)
#         col = re.sub(":","_",col)
#         col = re.sub("-","_",col)
#         array.append(col)
#     return array
# df.columns = columnsToString(df, columns)    
# print(df.columns)
# # for col in df.columns:
# #     print(dataUnderstand(df).col_Corr(col))
    
# # dport => There are Outlier Value
# df = df.drop(["]","UNKNOWN","(",],axis=1)
# #print(dataUnderstand(df).num_summary("dport",plot=True)) # before
# dataPreProcess(df).crush_outliers("dport",q1=0.25,q3=0.75) # %95:17, %100:35
# df.loc[df["dport"] > df["dport"].quantile(0.95), "dport"] = df["dport"].quantile(0.95)
# #print(dataUnderstand(df).num_summary("dport",plot=True)) # after
# targetDf = dataUnderstand(df).target_summary_with_cat_or_catNum("dport","Second")
# #print(targetDf)


# # for col in df.columns: # burada tum columns control edildi
# #     print(dataUnderstand(df).num_summary(col,plot=True)) # mac_or_request => outlier value
# #print(dataUnderstand(df).num_summary("mac_or_request",plot=True)) # before
# dataPreProcess(df).crush_outliers("mac_or_request",q1=0.25,q3=0.75)
# #print(dataUnderstand(df).num_summary("mac_or_request",plot=True)) # after


# #print(df.columns)

# # X_train,X_test,y_train, y_test = MachineLearning(df).get_dataset("TR_IST_AP_3",0.1,123)
# # X_train, X_test = MachineLearning(df).standardScaler(X_train, X_test)
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import cross_val_score
# # from sklearn.metrics import confusion_matrix,accuracy_score
# # lr = LogisticRegression()
# # lr.fit(X_train, y_train)
# # y_pred = lr.predict(X_test)
# # cv_result = cross_val_score(lr, X_train, y_train, scoring="accuracy",cv=10)
# # cv_mean = cv_result.mean()
# # acc_score = accuracy_score(y_test, y_pred)
# # sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt=".2f",cmap="Greens")
# # plt.title("Log Reg:{}".format(acc_score))
# # plt.show()


# # resultDf = MachineLearning(df).score("protocol=tcp",0.3,123)
# # #print("Model: {}\nCv Mean: {}\nAccuracy Score: {}".format(model,cv_mean,acc_score))
# # print(resultDf)

# '''
#                         cv_mean  acc_score
# LogisticReg            0.999714   1.000000
# KNN                    0.944714   0.948000
# SupportVectorMachines  0.757143   0.753000
# DecisionTree           0.999857   1.000000
# RandomForest           0.999857   1.000000
# Adaboost               1.000000   1.000000
# GradientBoost          0.999857   1.000000
# XGBoost                1.000000   1.000000
# DecisionTree           0.999857   1.000000
# LightGBM               0.999857   0.999667
# CatBoost               0.999857   1.000000
# '''
# # from sklearn.ensemble import AdaBoostClassifier
# # adaboost = AdaBoostClassifier()
# # pre_score,fScore, test_summary, train_summary = MachineLearning(df).get_model(adaboost,"TR_IST_AP_3",0.1,123,"accuracy",10)
# # print("f1_score:\n",fScore,"\nprecision_score: \n",pre_score,
# #       "\nTest Summary:\n ",test_summary,"\nTrain Summary: \n",train_summary)


# # print(df["protocol=udp"].value_counts(),"\n\n")
# # print(df["protocol=icmp6"].value_counts())


# sns.jointplot(data=df, x="Second",y="mac_or_dst",color=("purple"))
# plt.show()







