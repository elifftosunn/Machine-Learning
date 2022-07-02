try:
    import os
    import pandas as pd
    import sys
    import io
    import pymongo
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    import datetime
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
    def __init__(self, settings = None):
        self.settings = settings   
        try:
            self.mclient = MongoClient(host = self.settings.host,
                                      port = self.settings.port,
                                      maxPoolSize = self.settings.maxPoolSize)
        except Exception:
            return "Database not created"
'''
  mclient = MongoClient("localhost",27017)
  db = mclient["database"]
  collection = mclient["database"]["collection"]
'''
     
class MongoInformation(object):
    def __init__(self, client, databaseName, collectionName):
        self.client = client
        self.databaseName = databaseName
        self.collectionName = collectionName
    def databaseNames(self):    
            return self.client.mclient.list_database_names()
    def collectionNames(self, DbName):      
        if DbName is None:
            return []
        else:
            return self.client.mclient[DbName].list_collection_names()
    def dataFindAll(self):
        return self.client.mclient[self.databaseName][self.collectionName].find({})
    def dataFindQuery(self,query,columns = None):
        try:
            return self.client.mclient[self.databaseName][self.collectionName].find(query,columns)
        except Exception as e:
            return e
    def dataFindHead(self,limit):
        return self.client.mclient[self.databaseName][self.collectionName].find().limit(limit)
    def dataFindOne(self,query):
        try:
            return self.client.mclient[self.databaseName][self.collectionName].find_one(query)
        except Exception as e:
            return e
    def countData(self,query):
        result = self.client.mclient[self.databaseName][self.collectionName].count_documents(query)
        if result > 0:
            return result
        else:
            return "Not query!"
   
class MongoInsert(object):
    def __init__(self, _client = None, _mongoInformation = None):
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
    def __init__(self, _mongoInsert = None,_query = None, _newValues = None):
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
        #self._update = MongoUpdate(_mongoInsert = self._insert,_query= _query,_newValues = _newValues)
        self._delete = MongoDelete(self._client,self._mongoInformation)
    def insert_one(self, record):
        return self._insert.insert_One(record)
    def update(self,query,newValues):
        return MongoUpdate(self._insert,query,newValues) 
        #self._update = mongoUpdate(_mongoInsert = self._insert, self._query = query, _newValues) 
    def delete(self):
        return MongoDelete(self._client, self._mongoInformation)

def main():
    mongoResult =  MongoResult(host="localhost",
                               port=27017,
                               maxPoolSize=50,
                               databaseName="wifiData",
                               collectionName=("wifiCollection"))

    #print(mongoResult._mongoInformation.databaseNames())
    #print(mongoResult._mongoInformation.collectionNames("wifiData"))
    # for value in mongoResult._mongoInformation.dataFindAll():
    #     print(value)
    
    # for value in mongoResult._mongoInformation.dataFindQuery({"Server":"server-176.53.2.142.as42926.net"}):
    #     print(value)
    # print(mongoResult._mongoInformation.dataFindOne({"Server":"server-176.53.2.142.as42926.net"}))
    #print(mongoResult._mongoInformation.dataFindOne({"_id":ObjectId("62be8c87b56e0328f9229a48")}))
    # print(mongoResult._mongoInformation.countData({"Server":"server-176.53.2.142.as42926.net"}))
    # print(mongoResult._mongoInformation.countData({"flow_or_url":"flows"}))
    # for flow in mongoResult._mongoInformation.dataFindAll():
    #     print(flow,"\n")
    
    import dateutil
    # print(mongoResult._mongoInformation.countData({"protocol":"protocol=tcp"})) # 2441
    # # dst ile başlayanları al =>  $regex":"^dst
    # query = {"SNAT_or_DNAT":{"$regex":"^dst"}}
    # result = mongoResult._mongoInformation.dataFindQuery(query)
    # count = mongoResult._mongoInformation.countData(query)
    # for value in result[0:5]:
    #     print(value,"\n")
    # print("Count: ",count)
    
    # query2 = {"Date":{"$dayOfMonth":"2022-06-26T21:02:30.000+00:00"}}
    # result2 = mongoResult._mongoInformation.dataFindQuery(query2)
    # print(mongoResult._mongoInformation.countData(query2))
    # for value in result2:
    #     print(value,"\n")
    
    
    # SORGUYA GORE BELIRLI KOLONLARI CEKME
    columns = {"_id":0,"flow_or_url":1,"allow_or_src":2,"protocol":3,"Date":4}
    result3 = mongoResult._mongoInformation.dataFindQuery({
                                                        "flow_or_url":"flows"
                                                        },columns)
        
    def dateHourMinuteSecond():
        for value in mongoResult._mongoInformation.dataFindHead(5):
            # print(value["Date"])
            # print(type(value["Date"]))
            print(value["Date"].hour," ",value["Date"].minute," ",value["Date"].second)
    #dateHourMinuteSecond()
   
    
    class Question(object): # CLASS'A DONUSTURRRRRRRRRRRRRRRRRRRRRRRRR
        pass
    def SecondOrMinute(columns,query,second,minute):
        FeatureResult = mongoResult._mongoInformation.dataFindQuery(query,columns)    
        countSecond, countMinute = 0, 0
        for value in FeatureResult:
            result = value["Date"].second == second
            if result == True:
                countSecond +=1
        FeatureResult = mongoResult._mongoInformation.dataFindQuery(query,columns)    
        for value in FeatureResult:
            result = value["Date"].minute == minute
            if result == True:
                countMinute += 1
        return countSecond, countMinute
    # kac tane disaridan dst geliyor? kac saniyede geliyor ve en cok kacıncı saniyede geliyor?
    def dstResult():
        columns = {"_id":0,"mac_or_dst":1,"Date":2,"protocol":3}
        queryDst = {"mac_or_dst":{"$regex":"^dst"}}
        countResultDstSecond, countResultDstMinute = {}, {}
        for i in range(60):
            countSecond, countMinute = SecondOrMinute(columns,queryDst,i,i)
            countResultDstSecond[i] = countSecond
            countResultDstMinute[i] = countMinute
        # for key,value in countResultMinute.items():
        #     print(key,":",value)
        # for key,value in countResultSecond.items():
        #     print(key,":",value) 
        return countResultDstMinute,countResultDstSecond
    
    # HANGI PROTOCOL ILE GELIYOR VE PROTOCOL'UN GELME SURELERI NEDIR?
    df = pd.read_csv("datas\processedDatas\mongo-wifi.csv")
    #print(df["protocol"].value_counts())
    def protocolUdp():
        queryProtocolUdp = {"protocol":"protocol=udp"}
        columns = {"_id":0,"protocol":1,"Date":2}
        countResultUdpSecond, countResultUdpMinute = {}, {}
        for i in range(60):
            countSecond, countMinute = SecondOrMinute(columns,queryProtocolUdp,i,i)
            countResultUdpSecond[i] = countSecond 
            countResultUdpMinute[i] = countMinute 
        return countResultUdpMinute,countResultUdpSecond
    #protocolUdp()
    def protocolTcp():
        queryProtocolTcp = {"protocol":"protocol=tcp"}
        columns = {"_id":0,"protocol":1,"Date":2}
        countResultTcpSecond, countResultTcpMinute = {}, {}
        for i in range(60):
            countSecond, countMinute = SecondOrMinute(columns,queryProtocolTcp,i,i)
            countResultTcpSecond[i] = countSecond 
            countResultTcpMinute[i] = countMinute 
        return countResultTcpMinute,countResultTcpSecond
    #protocolTcp()
    def Results(*results):
        max_value_minute,max_key_minute = int(),int()
        for key,value in results[0].items():
            if value > max_value_minute:
                max_value_minute = value
                max_key_minute = key
        max_value_second,max_key_second = int(),int()
        for key,value in results[1].items():
            if value > max_value_second:
                max_value_second = value
                max_key_second = key        
        print("Max Value Minute: ",max_key_minute)
        print("Max Value Second: ",max_key_second)  
    countResultMinute,countResultSecond = dstResult() # Max Value Minute: 19, Max Value Second: 50
    countResultUdpMinute,countResultUdpSecond = protocolUdp() # Max Value Minute: 19, Max Value Minute: 50
    countResultTcpMinute,countResultTcpSecond = protocolTcp() # Max Value Minute: 19, Max Value Second: 56
    Results(countResultTcpMinute,countResultTcpSecond)
    
    
    
if __name__ == "__main__":  
    main()
    
