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
print(dataUnderstand(df))
df_copy = df.copy()

# FEATURE ENGINEERING
# - kac tane disaridan dst geliyor ve kac saniyede geliyor


def separatingColon(df, col, split, sep1, sep2):
    firstValues, secondValues = [], []
    firstCount, secondCount = 0, 0
    for col in df[col].values:
        col = str(col)
        values = col.split(split)
        if values[0] == sep1:
            if values[1] == None:
                values[1] = 1
                firstValues.append(values[1])
                firstCount += 1
            else:
                firstValues.append(values[1])
                firstCount += 1
        if values[0] == sep2:
            if values[1] == None:
                values[1] = 1
                secondValues.append(values[1])
                secondCount += 1
            else:
                secondValues.append(values[1])
                secondCount += 1
    return firstCount, secondCount, firstValues, secondValues


dstCount, macCount, dstValues, macValues = separatingColon(
    df, "mac_or_dst", "=", "dst", "mac")
print("dst count: ", dstCount, "\nmac count: ", macCount)
print(df["mac_or_dst"].describe())
dstDf = pd.DataFrame(data=dstValues, columns=["DST_MAC"])
macDf = pd.DataFrame(data=macValues, columns=["MAC"])
df = pd.concat([df, dstDf, macDf], axis=1)
df["DST_MAC"] = df["DST_MAC"].fillna(0)
df["MAC"] = df["MAC"].fillna(0)
df["MAC"] = df["MAC"].astype("str")
print(df["DST_MAC"].describe())
print(df["MAC"].describe())
# Categoric Columns Visualisation
dataUnderstand(df).categoricVisualisationBarplot("MAC", "Second")
dataUnderstand(df).categoricVisualisationCountplot("MAC")
dataUnderstand(df).categoricVisualisationBoxplot(
    "MAC", "Second")  # outlier values convert to number
# dataUnderstand(df).categoricVisualisationBoxplot("DST","Second") # outlier values convert to number(nunique value very much)
# df.to_csv("datas/processedDatas/data.csv")


# QUESTION 1- hangi protocol ile geliyor ve protocolun gelme sureleri nedir(tcp-udp ayrı ayrı)
src_Count, dst_src_Count, srcValues, dst_src_values = separatingColon(
    df, "SNAT_or_DNAT", "=", "src", "dst")
print("SRC Count: ", src_Count, "\nDST Count: ", dst_src_Count)
dst_src_Df = pd.DataFrame(data=dst_src_values, columns=["DST_SRC"])
src_Df = pd.DataFrame(data=srcValues, columns=["SRC"])
df = pd.concat([df, dst_src_Df, src_Df], axis=1)
# df["DST_SRC"] = df["DST_SRC"].fillna(0)
# df["SRC"] = df["SRC"].fillna(0)
print(df["DST_SRC"].describe(), "\n", df["SRC"].describe())
'''
- Buradan cikarilan sonuclara gore SOURCE NAT(SNAT) 10000 data'da 172.19.0.173 id'li ic agda bulanan pc
950 kez dıs agdaki pc net baglantisina erismis
- DESTINATION NAT(DNAT) 146.112.255.155:443 ip'li dis agda bulunan pc 332 kez ic agdaki ana pc netine erismis
'''
df["DST_SRC"] = df["DST_SRC"].fillna(0)
df["DST_SRC"] = df["DST_SRC"].astype("str")
df["SRC"] = df["SRC"].fillna(0)
df["SRC"] = df["SRC"].astype("str")

dataUnderstand(df).categoricVisualisationBarplot("SRC", "Second")
dataUnderstand(df).categoricVisualisationCountplot("SRC")
dataUnderstand(df).categoricVisualisationBoxplot("SRC", "Second")


mac_request_count, request_count, mac_requestValue, requestValue = separatingColon(
    df, "mac_or_request", "=", "mac", "request")
print("MAC Count: ", mac_request_count, "\nRequest Count: ", request_count)
mac_requestDf = pd.DataFrame(data=mac_requestValue, columns=["MAC_Req"])
df = pd.concat([df, mac_requestDf], axis=1)
print(df["MAC_Req"].describe())
df["MAC_Req"] = df["MAC_Req"].fillna(0)
df["MAC_Req"] = df["MAC_Req"].astype("str")
dataUnderstand(df).categoricVisualisationBarplot("MAC_Req", "Second")
dataUnderstand(df).categoricVisualisationCountplot("MAC_Req")
dataUnderstand(df).categoricVisualisationBoxplot("MAC_Req", "Second")
'''
Buradan cikarilan sonuclara gore 10000 data'da en fazla 950 kez F4:46:37:8A:F5:A7 id'li pc
MAC Pc'yi gecici yonlendirici olarak kullanmistir. Bu sekilde wifi'ye erismistir.
'''
sportCount, httpsCount, sportValue, httpValue = separatingColon(
    df, "sport", "=", "sport", "https")
print("Sport Count: ", sportCount, "\nHttp Count: ", httpsCount)
sportDf = pd.DataFrame(data=sportValue, columns=["SPORT"])
df = pd.concat([df, sportDf], axis=1)
print(df["SPORT"].describe())
'''
UDP paketinin hizmet adını, bağlantı noktası numarasını veya bağlantı noktası numarası aralığını kullanarak
hedef bağlantı noktasını belirtir.
En fazla hedef baglanti noktasini belirten port with 21 freq 5353 portudur.
'''
df["SPORT"] = df["SPORT"].fillna(0)
df["SPORT"] = df["SPORT"].astype("int32")
# There are outlier values
print(dataUnderstand(df).catchOutliers("SPORT", plot=True))
# Outlier values were discarded
dataPreProcess(df).crush_outliers("SPORT", q1=0.25, q3=0.75)
df.loc[df["SPORT"] < df["SPORT"].quantile(0.25), "SPORT"] = df["SPORT"].quantile(0.25) # ////////////////7
# Data after outliers discarded
print(dataUnderstand(df).catchOutliers("SPORT", plot=True))
sns.lineplot(data=df, x="Second", y="SPORT", hue="flow_or_url")
plt.legend(loc="lower right", shadow=True)
plt.show()


dportCount, noneCount, dportValues, noneValues = separatingColon(
    df, "dport", "=", "dport", None)
print("Dport Count: ", dportCount, "\nNone Count: ", noneCount)
dportDf = pd.DataFrame(data=dportValues, columns=["DPORT"])
df = pd.concat([df, dportDf], axis=1)
print(df["DPORT"].describe())
'''
UDP paketinin kaynak portunu, servis adını, port numarasını veya port numarası aralığını kullanarak belirtir.	
Veri gonderimindeki en fazla kullanılan kaynak port:53, 10000 datada 4472 frekans ile(nunique:25,count:7776)
Geriye kalan 2224 tanesi modemin portu herhangi bir yere yonlenmis oldugundan pasif olarak gelmistir.
'''
df["DPORT"] = df["DPORT"].fillna(0)
df["DPORT"] = df["DPORT"].astype("int32")
print(dataUnderstand(df).catchOutliers("DPORT", plot=True))
# Outlier values were discarded
dataPreProcess(df).crush_outliers("DPORT", q1=0.25, q3=0.75)
df.loc[df["DPORT"] > df["DPORT"].quantile(0.95), "DPORT"] = df["DPORT"].quantile(0.95) # ///////////////////////
df.loc[df["DPORT"] < df["DPORT"].quantile(0.25), "DPORT"] = df["DPORT"].quantile(0.25)
print(dataUnderstand(df).catchOutliers("DPORT", plot=True))






# # QUESTION 2: flows(net baglantisiz) and url(net baglantili) saniyede kac tane geliyor ve hangisi daha hizli
# flowTimeDf = df.loc[(df["flow_or_url"] == "flows"), [
#     "flow_or_url", "Second", "Minute"]]
# urlTimeDf = df.loc[(df["flow_or_url"] == "urls"), [
#     "flow_or_url", "Second", "Minute"]]
# flowDf = df.query('flow_or_url == "flows" and Minute == 2 and Second == 26')

# # print(df["Second"].describe(),"\n\n",df["Minute"].describe())
# def fastestCols(df):
#    values = []
#    max = 0
#    for i in range(60):
#        for j in range(43):
#             queryDf = df.query(f'flow_or_url == "flows" and Second == {i} and Minute == {j}')
#             values.append(queryDf)
#    for dataFrame in values:
#        if len(dataFrame) > max:
#            max = len(dataFrame)
#    for dataFrame in values:
#         if len(dataFrame) == max:
#             fastestFlowDf = dataFrame
#    return values,max,fastestFlowDf

# values,max,fastestFlowDf = fastestCols(df)
# print("MAX DataFrame Length as Flows: ",max) # 42
# '''
# 34th minute 50th second'de fastest internet baglantisi kullanilmadan Bluetooth veya 
# P2P Wi-Fi ile uygulamaya giris yapilmis
# '''
# def urlsCols(df):
#    valuesUrls = []
#    max = 0
#    for i in range(60):
#        for j in range(43):
#             queryDf = df.query(f'flow_or_url == "urls" and Second == {i} and Minute == {j}')
#             valuesUrls.append(queryDf)
#    for dataFrame in valuesUrls:
#        if len(dataFrame) > max:
#            max = len(dataFrame)
#    for dataFrame in valuesUrls:
#         if len(dataFrame) == max:
#             fastestUrlDf = dataFrame
#    return valuesUrls,max,fastestUrlDf
# valuesUrls,max,fastestUrlDf = urlsCols(df)
# print("MAX DataFrame Length as Urls: ",max) # 14
# '''
# 39th minute 50th second fastest net baglantisi kullanılarak uygulamaya giris yapilmis
# '''










# QUESTION 3: KABLOSUZ AGDAN HEDEFE KAC SANIYEDE GIDILIYOR YA DA SANIYEDE KAC KEZ GIDILEBILIYOR?
# SOURCE'DAN HEDEFE ARADA KAC SANIYE GECTIGINI BUL


# def source(df,value):
#     valueSources = []
#     max = 0
#     for i in range(60):
#         for j in range(43):
#             query = df.query(f'SRC == value and Second == {i} and Minute == {j}')
#             valueSources.append(query)
#     for dframe in valueSources:
#         if len(dframe) > max:
#             max = len(dframe)
#     for dframe in valueSources:
#         if len(dframe) == max:
#             fastestDf = dframe 
#     return valueSources, max, fastestDf
# valueSourcesDFrames, maxValues, fastestDFrames = [],[],[]
# for value in df["SRC"].unique():
#     val = '"'+str(value)+'"'
#     valueSources, max, fastestDf = source(df,val)
#     valueSourcesDFrames.append(valueSources)
#     maxValues.append(max)
#     fastestDFrames.append(fastestDf)


# DATA PREPARATION
datetime(df, "Time")
df = df.drop("Time",axis=1)

for col in df.columns:
    print(col,":\n",df[col].value_counts())

df = df.drop("DST_MAC",axis=1)
df = df.drop("MAC",axis=1)
df = df.drop("DST_SRC",axis=1)
df = df.drop("SRC",axis=1)
df = df.drop("MAC_Req",axis=1)

categoric_cols,numeric_cols,categoric_but_numeric = dataUnderstand(df.drop("ID",axis=1)).features()
print("categoric_cols: ",categoric_cols,"\nnumeric_cols",numeric_cols,"\ncategoric_but_numeric: ",categoric_but_numeric)


# Numeric Columns Visualisation
dataUnderstand(df).numericVisulisation(numeric_cols)

for col in df.columns:
    print(dataUnderstand(df).col_Corr(col))
'''
Maximum flows yani herhangi internet baglantisina ihtiyac duymadan yalnizca bluetooth veya P2P Wi-Fi ile uygulamaya giris yaparken diger kullanicilar ile mesajlasabiliyorsunuz
Maximum src yani SNAT: "ic" agdaki birden cok ana pc'nin "dis" agdaki herhangi bir ana pc'ye erismesine izin verir.
Maximum TR_IST_AP_4
Maximum protocol=udp yani veri gonderir ve alır, hızlıdır ama veri gonderimini garanti etmez
Maximum sport: https://api.opendns.com/... yani source port
Maximum dport: 53 yani hedef(destination) port

'''


# Categoric Columns Visualisation
# dataUnderstand(df).categoricVisualisationBarplot(categoric_cols,"Second")
# dataUnderstand(df).categoricVisualisationCountplot(categoric_cols)
# dataUnderstand(df).categoricVisualisationBoxplot(categoric_cols,"Second")
# dataUnderstand(df).categoricVisualisationStripplot(categoric_cols,"Second")
# dataUnderstand(df).categoricVisualisationViolinplot(categoric_cols,"Second")
# dataUnderstand(df).categoricVisualisationCountplot(categoric_but_numeric)


# Missing Value Control and Fillna
NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
print("NaN_Columns: ",NaN_Columns,"\n",missingDf)
# NaN_Columns:  ['protocol', 'sport', 'dport']
for col in NaN_Columns:
    missingValue(df).categoric_Freq(col)
NaN_Columns,missingDf = dataUnderstand(df).missingValueTables()
print("NaN_Columns: ",NaN_Columns,"\n",missingDf)



    
# categoric cols => numeric col
df = df.drop("ID",axis=1)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
for col in categoric_cols:
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(df[[col]])
    transformed_array = transformed.toarray()
    newDf = pd.DataFrame(transformed_array, columns=ohe.categories_)
    df = df.drop(col,axis=1)
    df = pd.concat([df,newDf], axis=1)
    print(df.shape) 
for col in categoric_but_numeric:
    df[col] = preprocessor.fit_transform(np.array(df[col]).reshape(-1,1))
    print(df.shape)
for col in categoric_but_numeric:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(df.shape)

print(df.columns)
# correcting the structure of the column
import re
def columnsToString(df,columns):
    array = []
    for col in df.columns:
        col = "".join(col)
        col = re.sub("/","_",col)
        col = re.sub(":","_",col)
        col = re.sub("-","_",col)
        array.append(col)
    return array
df.columns = columnsToString(df, columns)
print(df.columns)


# dport => There are Outlier Value
df = df.drop(["]","UNKNOWN","(",],axis=1)
print(dataUnderstand(df).num_summary("dport",plot=True)) # before
dataPreProcess(df).crush_outliers("dport",q1=0.25,q3=0.75) # %95:17, %100:35
df.loc[df["dport"] > df["dport"].quantile(0.95), "dport"] = df["dport"].quantile(0.95)
print(dataUnderstand(df).num_summary("dport",plot=True)) # after
targetDf = dataUnderstand(df).target_summary_with_cat_or_catNum("dport","Second")
print(targetDf)


# for col in df.columns: # burada tum columns control edildi
#      print(dataUnderstand(df).num_summary(col,plot=True)) # mac_or_request => outlier value
# print(dataUnderstand(df).num_summary("mac_or_request",plot=True)) # before
for col in df.columns:
    dataPreProcess(df).crush_outliers(col,q1=0.25, q3 = 0.75)
# for col in df.columns:
#     print(dataUnderstand(df).checkOutlier(col,q1=0.25,q3=0.75))
# for col in numeric_cols:
#     dataUnderstand(df).num_summary(col,plot=True)
# for col in categoric_but_numeric:
#     print(dataUnderstand(df).target_summary_with_cat_or_catNum(col,"Second"))
for col in categoric_cols:
    print(dataUnderstand(df).target_summary_with_cat_or_catNum(col,"Second"))

# dataPreProcess(df).crush_outliers("request_",plot=True, q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("mac_or_request",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("protocol=tcp",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("protocol=icmp6",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("protocol=2",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("protocol=0",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("POST",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("urls",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("flows",q1=0.25,q3=0.75)
# dataPreProcess(df).crush_outliers("events",q1=0.25,q3=0.75)



#print(df.columns)

# X_train,X_test,y_train, y_test = MachineLearning(df).get_dataset("TR_IST_AP_3",0.1,123)
# X_train, X_test = MachineLearning(df).standardScaler(X_train, X_test)
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix,accuracy_score
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# y_pred = lr.predict(X_test)
# cv_result = cross_val_score(lr, X_train, y_train, scoring="accuracy",cv=10)
# cv_mean = cv_result.mean()
# acc_score = accuracy_score(y_test, y_pred)
# sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt=".2f",cmap="Greens")
# plt.title("Log Reg:{}".format(acc_score))
# plt.show()


# resultDf = MachineLearning(df).score("protocol=tcp",0.3,123)
# #print("Model: {}\nCv Mean: {}\nAccuracy Score: {}".format(model,cv_mean,acc_score))
# print(resultDf)

'''
                        cv_mean  acc_score
LogisticReg            0.999714   1.000000
KNN                    0.944714   0.948000
SupportVectorMachines  0.757143   0.753000
DecisionTree           0.999857   1.000000
RandomForest           0.999857   1.000000
Adaboost               1.000000   1.000000
GradientBoost          0.999857   1.000000
XGBoost                1.000000   1.000000
DecisionTree           0.999857   1.000000
LightGBM               0.999857   0.999667
CatBoost               0.999857   1.000000
'''
# from sklearn.ensemble import AdaBoostClassifier
# adaboost = AdaBoostClassifier()
# pre_score,fScore, test_summary, train_summary = MachineLearning(df).get_model(adaboost,"TR_IST_AP_3",0.1,123,"accuracy",10)
# print("f1_score:\n",fScore,"\nprecision_score: \n",pre_score,
#       "\nTest Summary:\n ",test_summary,"\nTrain Summary: \n",train_summary)


# print(df["protocol=udp"].value_counts(),"\n\n")
# print(df["protocol=icmp6"].value_counts())


sns.jointplot(data=df, x="Second",y="mac_or_dst",color=("purple"))
plt.show()