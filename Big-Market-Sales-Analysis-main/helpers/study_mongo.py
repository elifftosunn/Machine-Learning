from pymongo import MongoClient
from bson.objectid import ObjectId

# https://www.programiz.com/python-programming/datetime
# https://mongoing.com/docs/reference/operator/aggregation/minute.html
# https://www.mongodb.com/docs/manual/reference/method/Date/

db = MongoClient(host="localhost",port=27017)
db = db.SupermarketDatabase.SupermarketCollection
#print(db.list_collection_names())

# *********************************** FIND AND COUNT ********************************************
values = db.find({})
# for value in values:
#     print(value,"\n")

#query = {"_id":ObjectId("62bede54b56e0328f9244b4a")}
# query = {"$and":[{"Branch":"C"},{"Quantity":{"$lte":10}},{"Gender":"Female"}]}
# result = db.find(query)
# for i in result[0:5]:
#     print(i,"\n")

# columns = {"_id":0,"Tax 5%":1,"Quantity":2,"Total":3,"Branch":4}
# query = {"Quantity":{"$eq":5}}
# query2 = {"Quantity":{"$ne":5}}
# #result = db.find(query,columns)
# result = db.count_documents(query) # 102
# result2 = db.count_documents(query2) # 898
# print("Count: ",result)
# # for i in result[0:5]:
# #     print(i)

# SORUN VAR TEKRAR BAKKKKKKKKKKKKKKKK !!!!!
# query3 = {"Gender":{"$not":re.compile("female")}} # string column oluyor sadece
# result3 = db.find(query3,columns)
# for i in result3:
#     print(i,"\n")

columns = {"_id":0,"Gender":1,"Branch":2,"City":3,"Tax 5%":4,"Date":5}
query4 =  {"$and":[
    {"City":"Naypyitaw"},
    {"Branch":"C"},
    {"Gender":{"$not":"female"}}
    ]}
result4 = db.find(query4,columns)

# for i in result4:
#     print(i,"\n")
# #print(db.count_documents(query3))
# print(db.count_documents(query4))
# # re.compile("female")



# *************************************** TIME/DATETIME ***********************************************
#*************************************** BELLI TARIHTEN ONCEKILERI ALMAAAA ****************************
import re, datetime, dateutil

# for value in mongoResult._mongoInformation.dataFindHead(5):
#     #value["Date"] = value["Date"].strftime("%Y-%m-%d %H:%M:%S") # convert string to date
#     #value["Date"] = value["Date"].strftime("%H:%M:%S") # convert string to date
#     #value["Date"] = datetime.datetime.strptime(value["Date"], "%H:%M:%S") # convert date to string
#     #value["Date"] = datetime.datetime.strptime(value["Date"], "%Y-%m-%d %H:%M:%S") # convert date to string
#     # print(value["Date"])
#     # print(type(value["Date"]))
#     print(value["Date"].hour," ",value["Date"].minute," ",value["Date"].second)

            
# strptime => convert date to string
# strftime => convert string to date
# strDate = "2019-02-04T21:00:00.000+00:00"
# result = db.find({"Date":{"$lt":dateutil.parser.parse(strDate)}},columns)
# for i in result:
#     print(i,"\n")
# print(db.count_documents({"Date":{"$lt":dateutil.parser.parse(strDate)}}))


# *********************************************  TIMEDELTA    ************************************
def timedelta():
    from datetime import timedelta
    time = "Timestamp({ t: 0, i: 11 })"
    result = db.find({'Time':{'$lt':datetime.datetime.now(), '$gt':datetime.datetime.now() - timedelta(days=800)}})
    for i in result:
        print(i)

    query = {'created':{'$lt':datetime.datetime.now(), '$gt':datetime.datetime.now() - timedelta(hours=500)}}
    result = db.find(query)
    for i in result:
        print(i)



def get_age_range(person_collection,min_age,max_age):
    query ={"$and":[
                {"age":{"$gte":min_age}},
                {"age":{"$lte":max_age}}
                ]}
    people = person_collection.find(query).sort("age")
    
    
def project_columns(person_collection):# "_id":0 => hicbir belgenin id alanını vermeyin demek
# ilk adi sonra soyadi al demek
    columns={"_id":0,"first_name":1,"last_name":2} 
    people = person_collection.collectionNames({}, columns) # columns => sadece belirtilen sütunlar
    for person in people:
        print(person)
# ************************************* UPDATE ONE/MANY *********************************************

def update_sales_by_id(sales_id):
    from bson.objectid import ObjectId
    _id = ObjectId(sales_id)
    all_updates = {
            "$set":{"new_field":True},  # "new_field":True => birden fazla degisiklik icin (birden fazla alan adi degistirmek istiyorsak oraya koyabiliriz)
            "$inc":{"Quantity":1},      # alanlardaki degerleri arttirma
            "$rename":{"City":"city","Customer type":"customer_type"} # alan adi degistirme
        }
    db.update_one({"_id":_id}, all_updates)
    
    db.update_one({"_id":_id}, {"$unset":{"Gender":"Male"}})
    db.update_one({"_id":_id}, {"$set":{"Customer type":"Normal"}})
    db.update_one({"_id":_id}, {"$unset":{"Product line":""}}) # belirli alanı kaldırma
    

    db.update_many({"_id":_id}, {"$and":[
        {"$set":{"City":"Samsun"}},
        {"$set":{"Quantity":8}}
        ]})

    
#update_sales_by_id("62bede54b56e0328f9244b4b")
# query = {"$and":[{"Quantity":5},{"Gender":"Female"},{"Payment":"Ewallet"}]}
# ************************************ UPDATE MANY ***************************************************
result = db.update_many({"$or":[
    {"_id":ObjectId("62bede54b56e0328f9244b55")},
    {"_id":ObjectId("62bede54b56e0328f9244c57")}
    ]}, 
    {"$currentDate":{"date":True}},
    upsert=True
)
result2 = db.update_many({"$and":[
    {"_id":ObjectId("62bede54b56e0328f9244b55")},
    {"_id":ObjectId("62bede54b56e0328f9244c57")}
    ]}, 
    {"$set":{"Name":"Elif"}}
    )
result3 = db.update_many({"_id":ObjectId("62bede54b56e0328f9244b55")}, {"$set":{"Name":"Elif","Age":20,"Lesson":"Math"}})
result4 = db.update_many({"$or":[
    {"_id":ObjectId("62bede54b56e0328f9244b55")},
    {"_id":ObjectId("62bede54b56e0328f9244c57")}
    ]}, 
    {"$set":{"Age":19,"Lesson":"Object Oriented Programming"}}
    )
# COLLECTION FEATURES BELIRLI ALANI KALDIRMA (UNSET)
result5 = db.update_many({"_id":ObjectId("62bede54b56e0328f9244b55")}, {"$unset":{"Name": ""}})
result6 = db.update_many({"_id":ObjectId("62bede54b56e0328f9244b55")}, {"$unset":{"Lesson": "","Rating": ""}})
# ALAN ADLARINI DEGISTIRME ("NAME":"NEW NAME") (RENAME)
result7 = db.update_many({"_id":ObjectId("62bede54b56e0328f9244b55")},{"$rename":{"Date":"start date","Age":"age"}})
result8 = db.update_many({"_id":ObjectId("62bede54b56e0328f9244b55")}, {"$rename":{"Branch":"new branch",
                                                                                   "Gender":"new Gender",
                                                                                   "Unit price":"new Unit price"}})
#print(result8)

# *************************************** REPLACE ONE ***********************************************
def replace_one(sales_id): 
    _id = ObjectId(sales_id)
    new_doc = {  # Belirtilen id'de sadece bu ozellikler olur, diger ozellikler kaybolur, eger id varsa bile yenisi olusur
        "Gender": "New Gender",
        "City": "New City",
        "Unit price":"New Unit price"
        }
    db.replace_one({"_id":_id}, new_doc)
#replace_one("62bede54b56e0328f9244b49")

# ********************************************* DELETE *******************************************************
def delete_doc_by_id(sales_id):
    _id =ObjectId(sales_id)
    #db.delete_one({"_id":_id})
    db.delete_many({"_id":_id})
    
#delete_doc_by_id("62bede54b56e0328f9244ca0")

query = {"$or":[
    {"City":"Naypyitaw"},
    {"Payment":"Ewallet"}
    ]}

values = db.find(query)
for value in values:
    print(value,"\n")
print(db.count_documents(query))

query2 = {"$and":[
    {"Gender":"Female"},
    {"lte":{"Quantity":7}},
    {"gt":{"Quantity":4}},
    {"in":{"Customer type":"Normal"}}
    ]}

result9 = db.find(query2)
for value in result9:
    print(value,"\n")
    
resultProject = db.aggregate(
   [
     {
       "$project":
         {
           "year": { "$year": "$Date" },
           "month": { "$month": "$Date" },
           "day": { "$dayOfMonth": "$Date" },
           "hour": {" $hour": "$Date" },
           "minutes": { "$minute": "$Date" },
          " seconds": { "$second": "$Date" },
           "milliseconds": { "$millisecond": "$Date" },
           "dayOfYear": { "$dayOfYear": "$Date" },
           "dayOfWeek": {" $dayOfWeek": "$Date" },
           "week": { "$week": "$Date" }
         }
     }
   ]
)

for i in resultProject:
    print(i,"\n ")

# ******************************* RELATIONSHIP *************************************************
address = {
    "_id" : "62bede54b56e0328f9244b4a",
    "street":"Bay Street",
    "number":2706,
    "city":"San Francisco",
    "country":"United States",
    "zip":"94107"
    }


def add_address_embed(person_id, address):
    _id = ObjectId(person_id)
    db.update_one({"_id":_id}, {"$addToSet":{"addresses":address}})

def add_address_relationship(person_id,address):
    _id = ObjectId(person_id)
    
    address = address.copy()
    address["owner_id"] = person_id
    
    #db = db.production.address => production collectionuna verilen is'ye address kısmı eklendi
    db.insert_one(address)
    
    
add_address_relationship("62bede54b56e0328f9244b49", address)



