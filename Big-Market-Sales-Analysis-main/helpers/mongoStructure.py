try:
    import os
    import pandas as pd
    import sys
    import io
    import pymongo
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    print("All modules loaded")
except Exception as e:
    print(f'Error : {e}')
    
class Singleton(type):
    _instance = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(Singleton, cls).__call__(*args,**kwargs)
            return cls._instance[cls] #type yazmamın sebebi class olup olmadıgını donduruyor(type)

class Settings(metaclass = Singleton):
    def __init__(self, _host ,_port, _maxPoolSize = 50):
        self.host = _host
        self.port = _port
        self.maxPoolSize = _maxPoolSize
class Client(metaclass = Singleton):
    def __init__(self, _settings):
        self.settings = _settings   
        # try:
        self.mclient = MongoClient(host = self.settings.host,
                                      port = self.settings.port,
                                      maxPoolSize = self.settings.maxPoolSize)
        # except Exception:
        #     return "Database not created"
'''
  mclient = MongoClient("localhost",27017)
  db = mclient["database"]
  collection = mclient["database"]["collection"]
'''
     
class MongoInformation(object):
    def __init__(self, _client, _databaseName, _collectionName):
        self.client = _client
        self.databaseName = _databaseName
        self.collectionName = _collectionName
    def databaseNames(self):   
        print("MongoInformation :  list_database_names")
        return self.client.mclient.list_database_names()
    def collectionNames(self, DbName):      
        if DbName is None:
            return []
        else:
            return self.client.mclient[DbName].list_collection_names()
    def dataFindAll(self):
        return self.client.mclient[self.databaseName][self.collectionName].find({})
    def dataFindQuery(self,query):
        try:
            return self.client.mclient[self.databaseName][self.collectionName].find(query)
        except Exception as e:
            return e
    def dataFindHead(self,limit):
        return self.client.mclient[self.databaseName][self.collectionName].find().limit(limit)
   
class MongoInsert(object):
    def __init__(self, _client, _mongoInformation):
        self.client = _client
        self.mongoInformation = _mongoInformation
    def insert_One(self, record = {}):
        try:
            self.client.mclient[self.mongoInformation.databaseName][self.mongoInformation.collectionName].insert_One(record)
            return True
        except Exception as e:
            return e     
    def insert_Multiple(self, records = []):       
        try:
            self.client.mclient[self.mongoInformation.databaseName][self.mongoInformation.collectionName].insert_Multiple(records)
            return True
        except Exception as e:
            return e
    def insert_DataFrame(self, df):   
        try:
            self.client.mclient[self.mongoInformation.databaseName][self.mongoInformation.collectionName].insert_many(df.to_dict(), ordered = False)
            return True
        # ordered = "True" ise (varsayılan) dokümanlar sunucuya verilen sıraya göre seri olarak eklenecektir. Bir hata oluşursa, kalan tüm ekler iptal edilir.
        # ordered = "False" ise, belgeler sunucuya rastgele sırayla, muhtemelen paralel olarak eklenir ve tüm belge eklemeleri denenir.
        except Exception as e:
            return e
  
class MongoUpdate(object):
    def __init__(self, _mongoInsert,_query, _newValues):
        self.mongoInsert = _mongoInsert
        self.query = _query
        self.newValues = _newValues
    def update_One(self):
        return self.mongoInsert.client.mclient[self.mongoInsert.databaseName][self.mongoInsert.collectionName].update_one(self.query,self.newValues)
  
    def update_Many(self):
        return self.mongoInsert.client.mclient[self.mongoInsert.databaseName][self.mongoInsert.collectionName].update_many(self.query,self.newValues)
    
class MongoDelete(object):
    def __init__(self,_client, _mongoInformation):
        self.client = _client
        self.mongoInformation = _mongoInformation
    def delete_One_Or_Multiple(self,query):
        try:
            dataWay = self.client.mclient[self.mongoInformation.databaseName][self.mongoInformation.collectionName]
            data = dataWay.find(query)
            if data is None:
                return "The query wasn't found"
            else:
                if data.count == 1:
                    oneData = dataWay.delete_one(query)
                    return str(oneData) + " deleted."
                else:
                    multipleData = dataWay.delete_many(query)
                    return str(multipleData.deleted_count) + " documents deleted."
        except Exception as e:
            return e
    def delete_All_Data(self):
        datas = self.client.mclient[self.mongoInformation.databaseName][self.mongoInformation.collectionName].delete_many({})
        return str(datas.deleted_count) + " documents deleted."

class MongoResult(object):  
    def __init__(self, host, port, maxPoolSize = 50, databaseName=None, collectionName=None):
        self._settings = Settings(host, port, maxPoolSize)
        self._client = Client(self._settings)
        self._databaseName = databaseName
        self._collectionName = collectionName
        self._mongoInformation = MongoInformation(self._client, self._databaseName, self._collectionName)
        self._insert = MongoInsert(_client = self._client,  _mongoInformation = self._mongoInformation)
        #self._update = MongoUpdate(_mongoInsert = self._insert, _query, _newValues)
        self._delete = MongoDelete(self._client,self._mongoInformation)
    def insert_one(self, record):
        return self._insert.insert_One(record)
    def update(self,query,newValues):
        return MongoUpdate(self._insert,query,newValues) 
        #self._update = mongoUpdate(_mongoInsert = self._insert, self._query = query, _newValues) 
    def delete(self):
        return MongoDelete(self._client, self._mongoInformation)

def NumberFeature(FeatureResult,feature): # columns unique values
    featureList = []
    for value in FeatureResult:
        if value[feature] not in featureList:
            featureList.append(value[feature])
    return featureList 

def NumberFeatureTotal(FeatureResult,*features):
    totalFeatureList = []
    for listFeatures in list(features):
        values = [feature for feature in listFeatures if feature != list] 
        # print(values) #['Server', 'TR_IST_AP', 'SNAT_or_DNAT', 'mac_or_dst', 'mac_or_request', 'protocol', 'sport', 'dport']
        for feature in values:
            featureList = NumberFeature(FeatureResult, feature)
            totalFeatureList.append(featureList) # each one columns unique values


def featureExtracting(FeatureResult):
    resultDf = pd.read_csv("lastMinuteSecond.csv")
    resultDf = resultDf.drop("Unnamed: 0",axis=1)
    return resultDf

# def MinuteSecondNumbers(FeatureResult,_minute,_second, _server, _tr_ist_ap, flow_or_url, _allow_or_src, _snat_or_dnat, _mac_or_dst, _mac_or_request, _protocol, _sport, _dport):
#     count = 0
#     # FeatureResult = MongoResult("localhost",27017)._mongoInformation.dataFindQuery(query)
#     for value in FeatureResult:
#         if value["Date"].minute == _minute and value["Date"].second == _second and value["Server"] == _server and value["TR_IST_AP"] == _tr_ist_ap and value["flow_or_url"] == flow_or_url and value["allow_or_src"] == _allow_or_src and value["SNAT_or_DNAT"] == _snat_or_dnat and value["mac_or_dst"] == _mac_or_dst and value["mac_or_request"] == _mac_or_request and value["protocol"] == _protocol and value["sport"] == _sport and value["dport"] == _dport:
#             count += 1
#     return count
def MinuteSecondNumbers(FeatureResult,_minute,_second):
    count = 0
    # FeatureResult = MongoResult("localhost",27017)._mongoInformation.dataFindQuery(query)
    for value in FeatureResult:
        if value["Date"].minute == _minute and value["Date"].second == _second:
            # print("Count method First: ",count)
            count += 1
            # print("Count method Second: ",count)
    return count
#countNumberMinute,countNumberSecond,minuteDf,secondDf,resultDf = results()

#tenThousandsDf = pd.read_csv("onBinDataFlowsAllowWifi.csv")
#tenThousandsDf = tenThousandsDf.drop("Unnamed: 0",axis=1)
#df = pd.read_csv("datas/processedDatas/mongo-wifi.csv")
def main():
    try:
        mongoResult =  MongoResult(host="localhost",
                                    port=27017,
                                    maxPoolSize=50,
                                    databaseName="wifiData",
                                    collectionName="wifiCollection")
    except Exception as e:
        print(e)
        
    query = {"$and":[{"flow_or_url":"flows"},{"allow_or_src":"allow"}]}  # flow and allow query
    #query = {"$and":[{"flow_or_url":"urls"},{"allow_or_src":{"$regex":"^src"}}]}  # urls and src query
    # FeatureResult = mongoResult._mongoInformation.dataFindQuery(query)
    # HER MINUTE AND SECOND ICIN GELEN FEATURE'LARI CEK ve ONLARI GRUPLAYIP KUMELEME(KNN) YAP(SINYALIN EN SIK GELDIGI ZAMANDAN EN AZ GELDIGI ZAMANA DOGRU)
    # EN SIK GELDIGI ZAMANDAKI WIFI BILGILERINI CEK VE ONLARI CLUSTER'LARA AYIR(1-10)
    # - Yeni dataframe'e her kolonu cekmeye calis
    # - mongo yapısını duzelt
    
      
    countWatchMinute, countWatchSecond, resultList, serverList, tr_ist_ap_List, flow_or_url_List, allow_or_src_List, snat_or_dnat_List, mac_or_dst_List,mac_or_request_List, protocolList, sportList, dportList = [],[], [],[],[],[],[],[],[],[],[],[],[]
    countNumberMinute, countNumberSecond, countNumberServer = {},{},{}
    countMinuteNumber, countSecondNumber  = 0,0
    for i in range(2,44):
        for j in range(60):
            # i.minute  j.second'daki Server'i cek
            FeatureResult = mongoResult._mongoInformation.dataFindQuery(query)
            for feature in FeatureResult:
                #print("Feature[Date].minute First : ",feature["Date"].minute,"          ",i) 
                #print("Feature[Date].second First : ",feature["Date"].second,"           ",j)
                if feature["Date"].minute == i and feature["Date"].second == j:
                    FeatureResult = mongoResult._mongoInformation.dataFindQuery(query)
                    print("Feature[Date].minute : ",feature["Date"].minute,"          ",i) 
                    print("Feature[Date].second : ",feature["Date"].second,"           ",j)
                    #print("Minute: ",i," Second: ",j)
                    count = MinuteSecondNumbers(FeatureResult, i, j)  
                    #print("Minute: ",i," Second: ",j," Count: ",count)    
                    countWatchMinute.append(i)
                    countWatchSecond.append(j)  
                    #countNumberMinute[i] = count # 43-54:0, 43-55:0
                    #countNumberSecond[j]   = count # {"41":0}, {"42":3}, {"43":11} 
                    resultList.append(count) # HERE ADDED 
                    serverList.append(feature["Server"])
                    tr_ist_ap_List.append(feature["TR_IST_AP"])
                    flow_or_url_List.append(feature["flow_or_url"])
                    allow_or_src_List.append(feature["allow_or_src"])
                    snat_or_dnat_List.append(feature["SNAT_or_DNAT"])
                    mac_or_dst_List.append(feature["mac_or_dst"])
                    mac_or_request_List.append(feature["mac_or_request"])
                    protocolList.append(feature["protocol"])
                    #print("feature['sport']: ",feature["sport"])
                    sportList.append(feature["sport"])
                    dportList.append(feature["dport"]) 
    # for key,value in countNumberMinute.items():
    #     print(key,"minute: ",value)
    # for key,value in countNumberSecond.items():
    #     print(key,"second: ",value)
    minuteDf = pd.DataFrame(countWatchMinute,columns=["Minute"])
    secondDf = pd.DataFrame(countWatchSecond,columns=["Second"])
    valuesDf = pd.DataFrame(resultList,columns=["RepeatNumber"])
    serverDf = pd.DataFrame(serverList, columns=["Server"])
    tr_ist_ap_Df = pd.DataFrame(tr_ist_ap_List, columns=["TR_IST_AP"])
    flow_or_url_Df = pd.DataFrame(flow_or_url_List, columns=["flow_or_url"])
    allow_or_src_Df = pd.DataFrame(allow_or_src_List, columns=["allow_or_src"])
    snat_or_dnat_Df = pd.DataFrame(snat_or_dnat_List, columns=["snat_or_dnat"])
    mac_or_dst_Df = pd.DataFrame(mac_or_dst_List, columns=["mac_or_dst"])
    mac_or_request_Df = pd.DataFrame(mac_or_request_List, columns=["mac_or_request"])
    protocolDf = pd.DataFrame(protocolList, columns=["protocol"])
    sportDf = pd.DataFrame(sportList, columns=["sport"])
    dportDf = pd.DataFrame(dportList, columns=["dport"])

    resultDf = pd.concat([minuteDf,secondDf,valuesDf,serverDf,tr_ist_ap_Df,
                          flow_or_url_Df,allow_or_src_Df,snat_or_dnat_Df,mac_or_dst_Df,
                          mac_or_request_Df, protocolDf,sportDf, dportDf],axis=1)
    #resultDf.to_csv("urls_and_src_data.csv",index=False)
    # resultDf.to_csv("flows_and_allow_data.csv",index=False)
    resultDf.to_csv("onBinDataFlowAllowWifi.csv",index=False)
    #resultDf.to_csv("onBinDataUrls_Src.csv",index=False)
    
 

   
    
  
if __name__ == "__main__":
    main()
    
'''
- kac tane disaridan dst geliyor ve kac saniyede geliyor
- hangi protocol ile geliyor ve protocolun gelme sureleri nedir(tcp-udp ayrı ayrı)
     dst=172.18.14.96         2439
     dst=172.18.14.225        1758
     dst=208.67.222.222        558
     mac=F4:46:37:8B:5E:7F     452
     dst=195.175.39.39         269
     dst=52.109.8.20             1
     dst=130.211.34.183          1
     dst=13.107.6.171            1
     dst=52.98.206.114           1
     dst=146.112.48.82           1
     Name: mac_or_dst, Length: 530, dtype: int64
     
     # Max Value Dst:  dst=172.18.14.96 => 2439
     server-176.53.2.142.as42926.net Max Value Server: 7184
     19 Max Value Minute: 316
     50 Max Value Second: 219
     
     
    Max Value Minute:  19
    Max Value Second:  50
    Max Value Dst:  dst=172.18.14.96 => 2439
    Server Dst Second  2 : 102
    Server Dst Second  3 : 128
    Server Dst Second  4 : 103
    Server Dst Minute  38 : 221
    Server Dst Minute  39 : 308
    Server Dst Minute  40 : 179
    {"$and":[{"mac_or_dst":"dst=172.18.14.96"},{"protocol":"protocol=udp"}]} => MongoDb => all(tamamı) protocol=udp
    Bu sonuclara gore packet type olarak basic udp protocol'u kullanilmis ve sadece data
    gonderip almak icin kullanılan bir packet turudur. Advantage olarak hızlıdır but bir oturum
    olusturmadigi and data teslimini guarantee etmedigi anlamina gelir.
    - 10000 data uzerinde en fazla 2439 freq Destination Port
    - 10000 data uzerinde en cok 7184 freq server-176.53.2.142.as42926.net  Server
    - 10000 data uzerinde en cok wifi kullanimi 19th minute 50th second gerceklesmistir.
    - 10000 data uzerinde en cok 7876 freq flows(net baglantisiz:Bluetooth or P2P_Wi-Fi)  
    - 10000 data uzerinde en cok 5335 freq udp packet type kullanilmistir. Yani veri gonderir ama teslimini garanti etmez.(baglantisiz)
    - 10000 data uzerinde en cok 4472 freq dport=53 => udp paketinin source port
    
'''
'''
 Flows Minute  19 : 316, Flows Second  50 : 219
 Bu sonuclara gore 19th minute 316 freq, 50th second 219 freq ile net baglantisina ihtiyac duyulmadan
 yalnizce Bluetooth ya da P2P Wi-Fi ile uygulamaya giris yapilmis
'''
'''
Max Value Minute Tcp => 19 : 316, Max Value Second Tcp => 56 : 174
Max Value Minute Udp =>  19 : 316, Max Value Minute Udp =>  50 : 219
Bu sonuclara gore 
'''
    
    
     

        