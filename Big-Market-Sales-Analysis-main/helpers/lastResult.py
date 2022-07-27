import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# DATA FIREWALL LOG BILGILERI 
# destination ip'ye kac adet src id veya mac addressi geliyor. Eger 1 dakika veya daha az bir surede 
# birden fazla request geliyor ise bu bir anomaly detection'dır ve muhtemel olarak ransomware 
# saldirisina maruz kaliyor. 


# DATE AYARINI YAP
# 


df = pd.read_csv("datas/TotalDatas/urlsSrcDatas/totalSportListConcatDictUrls.csv")
df_Flows = pd.read_csv("datas/TotalDatas/FlowDatas/totalSportListConcatDictFlows.csv")
print(df.corr())
sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.show()


lastData = pd.read_csv("datas/TotalDatas/FlowDatas/totalurls.csv")
# lastData["snat_or_dnat"] = lastData["SNAT_or_DNAT"]
def everyforDstMacAddress(): 
    totalMacCountDict = {}
    dstCountDict = df["snat_or_dnat"].value_counts().to_dict()
    for dstKey,dstValue in dstCountDict.items():
        dstDf = df.loc[df["snat_or_dnat"] == dstKey]
        macCountDict = dstDf["mac_or_dst"].value_counts().to_dict() # her bir dst icin kac adet mac address bulundugu
        totalMacCountDict[dstKey] = macCountDict # her bir dst'nin baglantili oldugu mac addreslerinin toplamını bir listenin elemani olarak ekleme
        # for macKey,macValue in macCountDict.items():    
        # # print("Mac Key:  ",macKey,"\nMac Value: ",macValue)
        #     print(macValue)
    return totalMacCountDict
# totalMacCountDict = everyforDstMacAddress()

def columnValueCounts(df,column,newColumnName):
    urls = df[column].value_counts().to_dict() #converts to dictionary
    df[newColumnName] = df[column].map(urls) 
# columnValueCounts(df, "allow_or_src", "Count_Src")
# columnValueCounts(df_Flows, "snat_or_dnat", "Count_Flow_Src")

def countSrcIp(df,countColumn):
    totalSrcList = []
    for count in df[countColumn].unique():
        if count > 20:
            countDf = df.loc[df[countColumn] == count]
            totalSrcList.append(countDf)
    return totalSrcList
# print(df["snat_or_dnat"].value_counts()) 
def macAddressAllocation():  # mac address ayirma
    totalMacDstList = []
    columnValueCounts(df, "snat_or_dnat", "Dst_Count")
    for count in df["Dst_Count"].unique():
        everyDstDf = df.loc[df["Dst_Count"] == count] # every a destination ip icin
        columnValueCounts(everyDstDf, "mac_or_dst", "Mac_Count") # her farkli mac adresi icin count   
        for macCount in everyDstDf["Mac_Count"].unique():
            if macCount > 20:
                macCountDf = everyDstDf.loc[everyDstDf["Mac_Count"] == macCount]
                totalMacDstList.append(macCountDf) # her bir destination ip icin mac addresinin kac
        #adet bulundugu ayrı ayrı dataframeler seklinde yazildi.
    return totalMacDstList
# totalMacDstList = macAddressAllocation()
# totalMacDstDf = pd.concat(totalMacDstList)
# totalMacDstDf.to_csv("datas/TotalDatas/urlsSrcDatas/lastResult.csv",index=False)

# ---------------------------------------------------------------------------------------------
# Datayı gruplayabilirsin => Hour, Minute, Second da ekle tableau icin
# df_groupby = df[["snat_or_dnat","mac_or_dst","MacCountToDst","allow_or_src","sport"]].groupby(["mac_or_dst"]).max().sort_values("MacCountToDst",ascending=False)
def dstMacAllocation(df):  # RANSOMWARE SALDIRILARINA MARUZ KALAN MAC ADDRESLERI BURADA CEKILDI
    totalMacDstList = []
    for dst in df["snat_or_dnat"].unique():
        dstDf = df.loc[df["snat_or_dnat"] == dst]
        for mac in dstDf["mac_or_dst"].unique():
            macDstDf = dstDf.loc[dstDf["mac_or_dst"] == mac]
            totalMacDstList.append(macDstDf)
    for dataframe in totalMacDstList:
        dataframe["MacCountToDst"] = len(dataframe)
    return totalMacDstList
totalMacDstList = dstMacAllocation(lastData)
# totalMacDstList = dstMacAllocation(df)
totalMacDstDf = pd.concat(totalMacDstList)
# totalMacDstDfMax = totalMacDstDf.loc[totalMacDstDf["MacCountToDst"] > 20]
# totalMacDstDfMax.to_csv("datas/TotalDatas/urlsSrcDatas/lastResult.csv",index=False)

# totalDstMacList = dstMacAllocation()     
# totalMacDstDf = pd.concat(totalDstMacList)
        
# EN SON YAPTIKLARIM
# df = pd.read_csv("datas/TotalDatas/urlsSrcDatas/lastResult.csv")
def timeListCreate(df):
    totalTimeList = []
    for hour in df["Hour"].unique():
        for minute in df["Minute"].unique():
            for second in df["Second"].unique():
                timeDf = df.loc[(df["Hour"] == hour) & (df["Minute"] == minute) & (df["Second"] == second)]
                totalTimeList.append(timeDf)
    count = 0
    maxTimeList = []
    for dataframe in totalTimeList:
        if len(dataframe) > 20:
            maxTimeList.append(dataframe)
            count += 1
    print("Count:   ",count)
    return maxTimeList
# maxTimeList = timeListCreate(totalMacDstDf)
# maxTimeDf = pd.concat(maxTimeList)
# maxTimeDf.to_csv("datas/TotalDatas/urlsSrcDatas/lastResult.csv",index=False)
def lastTimeDataFameCreate(maxTimeDf):
    lastTimeList = []
    for value in maxTimeDf["NumberEntryToSite"].unique():
        lastMaxTimeDf = maxTimeDf.loc[maxTimeDf["NumberEntryToSite"] == value]
        if len(lastMaxTimeDf) > 5:
            lastMaxTimeDf["More_Count"] = len(lastMaxTimeDf)
            lastTimeList.append(lastMaxTimeDf)
    return lastTimeList
# lastTimeList = lastTimeDataFameCreate(maxTimeDf)
# lastDf = pd.concat(lastTimeList)
# lastDf.to_csv("datas/TotalDatas/urlsSrcDatas/lastResult.csv",index=False)
     


# BU KISIMDA HEP URLS ICIN BAKILDI

# # WHICH TIME INTERVAL IS THERE MORE DENSITY?
# plt.figure(figsize=(20,8))
# sns.histplot(data = df, x = "Hour", y = "NumberEntryToSite")
# plt.ylim(0,1000)
# plt.show()
# # print(df.loc[df["Hour"] == 11]) # not data 11-20 between hours 


# WHICH DO HOURS MORE COK ENTRY?
def entryVisualization():
    for entryValue in df["NumberEntryToSite"].unique():
        entryNumber = df.loc[df["NumberEntryToSite"] == entryValue]
        plt.figure(figsize=(15,8))
        sns.countplot(entryNumber["Hour"])
        plt.show()  

# WHICH DO MAC ADDRESS WHICH SITES MORE ENTRY?
def entrySitesMac():
    for mac in df["mac_or_dst"].unique():
        macDf = df.loc[df["mac_or_dst"] == mac]
        plt.figure(figsize=(20,10))
        sns.countplot(macDf["NumberEntryToSite"])
        plt.title(mac)
        plt.show()
        
# entrySitesMac()
# HER MAC ADDRESININ EN COK(MAX) GIRIS YAPTIGI SITELER
def everyMacAddressEntrySitesNumber():
    everyMacMaxSitesEntry = []
    for mac in df["mac_or_dst"].unique():
        maxMacDf = df.loc[df["mac_or_dst"] == mac]
        maxValue = max(maxMacDf["NumberEntryToSite"])
        sportDf = maxMacDf.loc[maxMacDf["NumberEntryToSite"] == maxValue]
        sport = sportDf["sport"].unique()
        newDf = pd.DataFrame({"Mac":mac,"Max_Value":maxValue,"Sport":sport})
        everyMacMaxSitesEntry.append(newDf)
    return everyMacMaxSitesEntry
# everyMacMaxSitesEntry = everyMacAddressEntrySitesNumber()
# everyMacMaxSitesEntryDataFrame = pd.concat(everyMacMaxSitesEntry)
# # everyMacMaxSitesEntryDataFrame = pd.DataFrame(list(everyMacMaxSitesEntry.items()), columns=["Mac","Number_Entry_Sites"],index=range(len(everyMacMaxSitesEntry)))
# everyMacMaxSitesEntryDataFrame.to_csv("datas/TotalDatas/urlsSrcDatas/everyMacMaxSitesEntryDataFrame.csv",index=False)



# HER MAC ADDRESSININ SITELERDE KAC SAAT KALDIKLARI? 
# sns.distplot(df["Hour"])
# plt.show()
# HER BIR SITEYE GIREN KISILERIN SAYISI VE MAC ADDRESSLERI(EN COK HANGI SAATLERDE GIRIS YAPILMIS)
# x = kisi sayısı 
# y = siteler
# SITELERE MAXIMUM GIRIS SAYISININ ILK 50'SI
def PersonNumberToSites():
    totalPersonList = []
    for sport in df["sport"].unique(): # every site
        sportDf = df.loc[df["sport"] == sport] # dataframe for every site
        print(sportDf["mac_or_dst"].unique())
        newDf = pd.DataFrame({"Sport":sport,"Person_Number":len(sportDf["mac_or_dst"].unique())})
        totalPersonList.append(newDf)
    return totalPersonList
#totalPersonList = PersonNumberToSites()
# newDataFrame = pd.concat(totalPersonList)
# newDataFrame.to_csv("datas/TotalDatas/personNumber.csv",index=False)
# newDataFrame = pd.read_csv("datas/TotalDatas/personNumber.csv")
# data = newDataFrame.sort_values("Person_Number",ascending=False)
# data.head(50).to_csv("datas/TotalDatas/personNumber.csv",index=False)

def hourEntrySites():
    totalHourDict = {}
    for hour in df["Hour"].unique():
        hourDf = df.loc[df["Hour"] == hour]
        print(hour,"   ",len(hourDf["sport"].unique()))
        totalHourDict[hour] = len(hourDf["sport"].unique())
    return totalHourDict
# totalHourDict = hourEntrySites()
# print(totalHourDict)
# totalHourDf = pd.DataFrame(list(totalHourDict.items()), columns=["Hour","Urls_Number"])
# totalHourDf.to_csv("datas/TotalDatas/urlsNumber.csv",index=False)
