import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from helpers.data_helper import *
import requests, json, re
from tabulate import tabulate

# barplot: kategorik => toplam, ortalama, medyan
# histogram: surekli values(numeric value dagilimi)
# kdeplot: numeric dataların ayrık olarak degilde surekli olarak dagilimini gorsellestirmek istedigimiz zaman kullaniriz
# Boxplot: ortası medyan degeri, kutunun basi ve sonu %25(alt ceyrek) and %75(ust ceyrek)'lik kismi olusturur, cizginin basi min, sonu max
# Medyan => Aykiri degerlerin cok oldugu durumlarda mean yerine koyariz

# RepaeatNumber => Kac kez o minute and second'de entry yapildigi
# df = pd.read_csv("onBinDataFlowsAllowWifi.csv")
# print(df["snat_or_dnat"].value_counts())
# print(df["snat_or_dnat"].unique())
# listSourceId = df["mac_or_dst"].unique()
# listSourceId = df["snat_or_dnat"].unique()
# for ip in listSourceId:
#     print("SRC Ip: ",ip[4:])

def ipQuery():
    #Sorgu sonucu listelenecek başlıklar:
    # print ("IP","City","İl","İlçe","Enlem","Boylam")
    #Sorgulanacak ip lere ait liste:
    # ip = ["176.55.55.252","176.55.55.252","88.230.177.68","88.230.177.68"]
    listSourceId = df["snat_or_dnat"].unique()
    print(listSourceId)
    APIKEY = "00a7024f126a38e23aa70db911478316"
    #ip listesinin teker teker web servis üzerinden sorgulanması:
    for x in listSourceId:
       print(x[4:])
       #Bu kısımda yer alan API_KEY'i https://ipstack.com/ adresine üye olarak temin edebilirsiniz.
       serviceURL = "http://api.ipapi.com/"+x[4:]+"?access_key="+APIKEY+"&output=json"  
       r = requests.get(serviceURL)
       data = json.loads(r.text)
       # print(y["ip"],y["country_name"],y["region_name"],y["city"],y["latitude"],y["longitude"])  
      
       table =   [["IP-Address     ",data["ip"]],
                  ["City           ",data["city"]],
                  ["Country        ",data["country_name"]],
                  ["Region         ",data["region_name"]],
                  ["Capital        ",data["location"]["capital"]]] # bolge
      
      
       print(tabulate(table))
#ipQuery()  


def sourceIdDestinationIdQuery():
    for port in df["snat_or_dnat"].unique():
        sourcePort = df[df["snat_or_dnat"] == port]
        print("{} source port ip'li kisinin destination ips: ".format(port))
        print(sourcePort["mac_or_dst"].value_counts())
        plt.style.use("ggplot")
        plt.figure(figsize=(20,6), dpi=300)
        sns.countplot(sourcePort["mac_or_dst"])
        plt.xticks(rotation=90)
        plt.title("{} sites that a person with a port ip visits the most".format(port))
        plt.show()
#sourceIdDestinationIdQuery()
df = pd.read_csv("onBinDataUrls_Src.csv")
# hangi sitelerde ne sıklıkta ne kadar kaldiklari
def macAddressMostVisitSites():  
    for mac in df["mac_or_dst"].unique():
        moreMac = df.loc[df["mac_or_dst"] == mac]
        print("Mac Adresli Kisinin Bilgileri: ",mac)
        print(moreMac["sport"].value_counts())
        plt.figure(figsize=(20,6), dpi=300)
        sns.countplot(moreMac["sport"])
        plt.xticks(rotation=90)
        plt.title("{} sites that a person with a mac address visits the most".format(mac))
        plt.show()
# macAddressMostVisitSites()
print(df["mac_or_dst"].unique())
# for every mac address every sport(ziyaret edilen sitelerin) which minute, second and repeat number 
def everyMacAddressSportQuery(mac):
    sportList = []
    OnemacAddressDf = df.loc[df["mac_or_dst"] == mac]
    for sport in OnemacAddressDf["sport"].unique():
        sportDf = OnemacAddressDf.loc[OnemacAddressDf["sport"] == sport]
        sportList.append(sportDf)
    return sportList
# everyMacAddressSportQuery all data uzerindeki mac address's sport address
def totalMacSportDF():
    totalSportList = {}
    for mac in df["mac_or_dst"].unique():
        sportList = everyMacAddressSportQuery(mac)
        totalSportList[mac] = sportList
    return totalSportList
        # her bir mac adresinin her bir site ziyareti(sport) dataframe olarak cekildi. => totalSportList

def everyMacCsvCreate():
    totalMacDf = []
    for mac in df["mac_or_dst"].unique():
        everyMacAddressDataFrame = df.loc[df["mac_or_dst"] == mac]
        totalMacDf.append(everyMacAddressDataFrame)
    
    # For Every Mac Address Csv File Create
    # count = 0
    # for df in totalMacDf:
    #     df.to_csv("datas/urls_Src_Datas/macDataFrame{}.csv".format(count+1),index=False)
    #     count += 1













# from urllib import response
# from ip2geotools.databases.noncommercial import  DbIpCity
# import socket
# url = input('Web site urlsi giriniz: ')
# ip = socket.gethostbyname(url)
# print('CODE HUB')
# response = DbIpCity.get(ip, api_key='free')
# print('İP ADRESİ: ', ip)
# print('ŞEHİR: ', response.city)
# print('ÜLKE: ', response.country)
# print('BÖLGE: ', response.region) 













# for column in flowAllowWifi.columns:
#     print(flowAllowWifi[column].value_counts(), "\n")
# plt.figure(figsize=(20, 8))
# sns.lineplot(data=flowAllowWifi, x="TR_IST_AP", y="Second")
# plt.xticks(rotation=60)
# plt.show()
# moreRepeat = flowAllowWifi.sort_values("RepeatNumber", ascending=False)
# results = moreRepeat[moreRepeat["RepeatNumber"] == 42]


# # def results(df, feature, *args):
# #     count = 1
# #     # print("Len Args: ",len(args))
# #     for i in range(len(args)-1):
# #         # print("args[i]: ",args[i])
# #         # print("args[i+1]: ",args[i+1])
# #         df.loc[(df[feature] >= args[i]) & (
# #             df[feature] < args[i+1]), feature] = count
# #         count += 1
# #     return df


# # df = results(flowAllowWifi, "RepeatNumber", 5, 12, 18, 24, 36, 45)
# print(dataUnderstand(flowAllowWifi))
# df = flowAllowWifi
# categoric_cols, numeric_cols, categoric_but_numeric = dataUnderstand(df).features()
# for col in categoric_cols:
#     print(dataUnderstand(df).target_summary_with_cat_or_catNum(col, "Second"))
# df = dataUnderstand(df).num_to_cat("RepeatNumber",5,12,18,24,36,45)
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import confusion_matrix,accuracy_score

# for col in categoric_but_numeric:
#     print(df[col].shape)
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
# for col in categoric_cols:
#     ohe = OneHotEncoder()
#     transformed = ohe.fit_transform(df[[col]])
#     transformed_array = transformed.toarray()
#     newDf = pd.DataFrame(data=transformed_array,columns=ohe.categories_)
#     df = df.drop(col,axis=1)
#     df = pd.concat([df,newDf],axis=1)
# X = df.drop("RepeatNumber",axis=1)
# y = df["RepeatNumber"]     

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=123)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# wcss = []
# for i in range(1,20):
#     kmeans = KMeans(n_clusters = i,init = 'k-means++', random_state = 42)
#     kmeans.fit_transform(X_train, y_train)
#     y_pred = kmeans.predict(X_test)
#     wcss.append(kmeans.inertia_)
    
# plt.plot(wcss)
# plt.show()
# kmeans = KMeans(n_clusters=4,init="k-means++")
# kmeans.fit_transform(X_train,y_train)
# y_pred = kmeans.predict(X_test)
# print("Accuracy Score: ",accuracy_score(y_test, y_pred))
# sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt=".2f")
# plt.title("KMeans Algorithm")
# plt.show()



# knn = KNeighborsClassifier(n_neighbors=2,metric="minkowski")
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print("Accuracy Score: ",accuracy_score(y_test, y_pred))
# sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt=".2f")
# plt.title("KNN Algorithm")
# plt.show()

# plt.figure(figsize=(20,5),dpi=300)
# # sns.countplot(data=newDf,x="RepeatNumber")
# # sns.scatterplot(data=newDf, x="Minute", y="RepeatNumber")
# plt.show()

# print(flowAllowWifi.describe())e
# plt.figure(figsize=(20,5),dpi=300)
# sns.countplot(flowAllowWifi["mac_or_request"])
# plt.xticks(rotation=60,fontsize=14)
# plt.title("Flow_Or_Allow  Mac_or_Request Count")
# plt.show()

# # print(len(flowAllowWifi.loc[flowAllowWifi["mac_or_request"] == "mac=F4:46:37:8A:F5:A7"]))
# '''
# in 10000 data more 950 freq (mac=F4:46:37:8A:F5:A7) mac address entry yapmis
# '''

# plt.figure(figsize=(20,5),dpi=300)
# sns.boxplot(data = flowAllowWifi, x = "mac_or_request", y = "RepeatNumber")
# plt.xticks(rotation=60,fontsize=14)
# plt.title("Flow_Or_Allow Wifi Repeat Number")
# plt.show()

# plt.figure(figsize=(20,5),dpi=300)
# sns.barplot(data = flowAllowWifi, x = "mac_or_request", y = "Minute") # => more around 20 minute
# plt.xticks(rotation=60,fontsize=14)
# plt.title("Flow_Or_Allow Wifi Minute by Mac_or_Request")
# plt.show()

# maxFlowAllow = flowAllowWifi.sort_values(["RepeatNumber"],ascending=False).head(42)
# print("maxFlowAllow:protocol=tcp => ",len(maxFlowAllow.loc[maxFlowAllow["protocol"] == "protocol=tcp"])) # => 10 tcp, 32 udp used
# # protocol => udp kullanildiginda daha fazla giris denemeleri oluyor
# print(maxFlowAllow["mac_or_request"])

# '''
# mac=F4:D4:88:8A:A4:63
# mac=F4:46:37:8B:43:EA
# mac=F4:46:37:8B:66:DB
# more repeat number in 10000 data, 34th minute and 50th second, 42 freq
# '''
# minFlowAllow = flowAllowWifi.sort_values("RepeatNumber",ascending=True).head(188)
# print(minFlowAllow["mac_or_request"].value_counts())
# '''
# mac=BC:09:1B:F0:66:C1    31
# mac=F4:46:37:8B:43:EA    28
# mac=F4:46:37:8A:F5:A7    20
# mac=F4:46:37:D8:6D:B9    20
# mac=20:3C:AE:E3:37:D0    17
# mac=F4:D4:88:8A:A4:63    14
# mac=F4:46:37:8B:43:36    12
# mac=BC:09:1B:F8:BA:51    12
# mac=4C:02:20:07:7B:B3    10
# mac=F4:46:37:8B:1F:28     7
# mac=4E:45:D9:75:C4:C9     7
# mac=BC:09:1B:DD:22:28     6
# mac=F4:46:37:8B:66:DB     4
# Name: mac_or_request, dtype: i

# mac=BC:09:1B:F0:66:C1 => in 10000 data different times 31 freq entry(arada seconds(1-2 second) olan da var)
# mac=F4:46:37:8B:43:EA => in 10000 data different times 28 freq entry
# '''

# print("flowAllowWifi least repeat number =>  ",len(flowAllowWifi.loc[flowAllowWifi["RepeatNumber"] == 1]))

# print("flowAllowWifi more repeat number =>  ",len(flowAllowWifi.loc[flowAllowWifi["RepeatNumber"] == 42]))

# moreMac = minFlowAllow.loc[minFlowAllow["mac_or_request"] == "mac=BC:09:1B:F0:66:C1"]

# print("minFlowAllow:protocol=tcp => ",len(minFlowAllow.loc[minFlowAllow["protocol"] == "protocol=tcp"])) # => 104 tcp, 84 udp used
# # protocol => tcp kullanildiginda daha az giris denemeleri oluyor.


# urlsSrcWifi = pd.read_csv("onBinDataUrls_Src.csv")

# plt.figure(figsize=(20,5),dpi=300)
# sns.countplot(urlsSrcWifi["mac_or_dst"])
# plt.xticks(rotation=60,fontsize=14)
# plt.title("urlsSrcWifi Mac_or_Dst Count")
# plt.show()

# plt.figure(figsize=(20,5),dpi=300)
# sns.barplot(data = urlsSrcWifi, x = "mac_or_dst", y = "RepeatNumber")
# plt.xticks(rotation=60,fontsize=14)
# plt.title("urlsSrcWifi Mac_or_Dst Repeat Number")
# plt.show()
# '''
# More Repeat Number
# '''

# plt.figure(figsize=(20,5),dpi=300)
# sns.boxplot(data = urlsSrcWifi, x = "mac_or_dst", y = "RepeatNumber")
# plt.xticks(rotation=60,fontsize=14)
# plt.title("urlsSrcWifi Mac_or_Dst Repeat Number Outlier Values")
# plt.show()

# maxUrlOrSrc = urlsSrcWifi.sort_values("RepeatNumber",ascending=False)
# '''
# More with 14(same second and same minute) repeat 27th minute 14th second and 39th minute 50th second'de entry yapilmis

# '''

# maxUrlsWifi = urlsSrcWifi.loc[urlsSrcWifi["RepeatNumber"] == 14]
# print(maxUrlsWifi["mac_or_dst"].value_counts())

# '''

# mac=4C:02:20:07:7B:B3    13
# mac=F4:46:37:8B:1F:28    12
# mac=BC:09:1B:F8:BA:51     1
# mac=20:3C:AE:E3:37:D0     1
# mac=F4:46:37:8B:43:EA     1

# This mac address'leri  14 freq(same second and same minute) 39th minute 50th second and 27th minute 14th second  giris yapilmis.
# More freq  mac=4C:02:20:07:7B:B3 address in all data 167 freq var and
# '''

# print(urlsSrcWifi["mac_or_dst"].value_counts())

# print(urlsSrcWifi["mac_or_dst"].value_counts())


# data = urlsSrcWifi.loc[urlsSrcWifi["mac_or_dst"] == "mac=BC:09:1B:DD:22:28"]
# print(data["RepeatNumber"].sort_values(ascending=False))








