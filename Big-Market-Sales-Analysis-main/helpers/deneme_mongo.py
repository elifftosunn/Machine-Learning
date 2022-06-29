from pymongo import MongoClient
import dns
import datetime
import pandas as pd
import csv,json


cluster = "mongodb+srv://new_user:12345@cluster0.pstweiz.mongodb.net/test?retryWrites=true&w=majority"
client = MongoClient(cluster)
print(client.list_database_names())

db = client.test # database erisildi
print(db.list_collection_names())
todo = {"name":"Patrick","text":"My first todo!",
         "status":"open",
         "tags":["Python","Coding"],
         "date":datetime.datetime.utcnow()}
todo1 = {"name":"Patrick","text":"My second todo!",
         "status":"open",
         "tags":["Python","Coding"],
         "date":datetime.datetime(2022,1,1,10,45)}
#db.todos.insert_one(todo) # todo added
#db.todos.insert_one(todo1) # todo1 added
todos2 = [
    {"name":"Elif","text":"My second todo!",
             "status":"open",
             "tags":["Spyder","Mongo"],
             "date":datetime.datetime(2022,3,5,10,30)},
    {"name":"Hatice","text":"My third todo!",
             "status":"open",
             "tags":["Notebook","Anaconda3"],
             "date":datetime.datetime(2022,1,1,10,45)}
    ]
#db.todos.insert_many(todos2) #todos2 added

result = db.todos.find_one({"name":"Patrick","text":"My second todo!"})
#print(result)
result2 = db.todos.find_one()
#print("\n",result2)
result3 = db.todos.find_one({"tags":"Notebook"})
#print(result3)
todo3 ={"name":"Selin","text":"My fourth todo!",
     "status":"open",
     "tags":["C#","ASP.Net","React.js"],
     "date":datetime.datetime.utcnow(),
     "lesson":["Object Oriented Programming","data structures and algorithms"],
     "subject":["Data Science","Machine Learning"]}
#db.todos.insert_one(todo3)
result4 = db.todos.find_one({"subject":"Data Science"})
#print(result4)


# finding it by id
from bson.objectid import ObjectId
result5 = db.todos.find_one({"_id":ObjectId("62b805e0ae8e032cab3a332b")})
#print(result5)

# # finding multiple data
# results = db.todos.find({"name":"Patrick"})
# results2 = db.todos.find({"name":"Patrick"})
# print("********* Results2***********")
# for result in results2:
#     print(result)
# print("********* Results***********")
# for person in list(results):
#     print(person)
  

# finding the number of data  
print(db.todos.count_documents({})) # data number
print(db.todos.count_documents({"tags":"Python"}))


print("***************DATETIME*********************")
# a date smaller than the current date
d = datetime.datetime(2022,2,2)
results3 = db.todos.find({"date":{"$lt":d}}).sort("name")
for result in results3:
    print(result)
# $lt => a date less than the current date
# #gt => a date greater than the current date


# delete data
result6 = db.todos.delete_one({"_id":ObjectId("62b80b92ae8e032cab3a334a")})
print(result6)
result7 = db.todos.delete_many({"name":"Patrick"})


# update data
result8 = db.todos.update_one({"tags":"C#"},{"$set":{"name":"Zeynep","status":"close"}})
print(result8)
# removing the key-value pair
result9 = db.todos.update_one({"tags":"Notebook"}, {"$unset":{"status":None}})
print(result9)
# if not key-value pair => update
result9 = db.todos.update_one({"tags":"Notebook"}, {"$set":{"status":"NEW OPEN"}})
print(result9)



# # convert-pandas-dataframe-to-json
# df = pd.read_csv("datas/Test.csv")
# df = pd.DataFrame(df)
# data = df.to_json(orient = "index")  # orient: index, columns, values, table
# parsed = json.loads(data)

# for key,value in parsed.items():
#     for val in value:
#         print(val)
# for key,value in parsed.items():
#     print(key)      
#print(data)
# jsonFilePath  = "datas/jsonData/json_test.json"
# with open(jsonFilePath,"w") as jsonFile:
#     jsonFile.write(json.dumps(data, indent=4))




# # convert-csv-to-json-with-python
# csvFilePath = "datas/Train.csv"
# jsonFilePath  = "datas/jsonData/json_train.json"
# data = {}
# with open(csvFilePath) as csvFile:
#     csvReader = csv.DictReader(csvFile)
#     for rows in csvReader:
#         id = rows["Item_Identifier"]
#         data[id] = rows
# with open(jsonFilePath,"w") as jsonFile:
#     jsonFile.write(json.dumps(data, indent=4))
    


