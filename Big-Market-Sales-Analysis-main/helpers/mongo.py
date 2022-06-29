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
    def dataFindQuery(self,query):
        try:
            return self.client.mclient[self.databaseName][self.collectionName].find(query)
        except Exception as e:
            return e
    def dataFindHead(self,limit):
        return self.client.mclient[self.databaseName][self.collectionName].find().limit(limit)
   
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
        #self._update = MongoUpdate(_mongoInsert = self._insert, _query, _newValues)
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
                               databaseName="denemeDtaBase",
                               collectionName=("denemeCollection"))
    record = {"name":"Elif","age":20}
    mongoResult.insert_one(record)
    print(mongoResult._mongoInformation.databaseNames())
    print(mongoResult._mongoInformation.collectionNames("denemeDatabase"))
    print(mongoResult) 

if __name__ == "__main__":
    main()
    
    

        