from helpers.mongoStructure import *

#     def __init__(self, host, port, maxPoolSize = 50, databaseName=None, collectionName=None):

mongoResult = MongoResult(host="localhost",port=27017,databaseName="wifiData",collectionName="wifiCollection")
# print(mongo._mongoInformation.databaseNames())
# results = mongo._mongoInformation.dataFindAll()
# for value in results:
#     print(value,"\n")
   
    
# KOD SATIRLARININ TAMAMINI SOLA KAYDIRMA => Ctrl + Alt Gr + [    
          
# query = {"$and":[{"flow_or_url":"flows"},{"allow_or_src":"allow"}]}  # flow and allow query
query = {"$and":[{"flow_or_url":"urls"},{"allow_or_src":{"$regex":"^src"}}]}  # urls and src query
# FeatureResult = mongoResult._mongoInformation.dataFindQuery(query)
# HER MINUTE AND SECOND ICIN GELEN FEATURE'LARI CEK ve ONLARI GRUPLAYIP KUMELEME(KNN) YAP(SINYALIN EN SIK GELDIGI ZAMANDAN EN AZ GELDIGI ZAMANA DOGRU)
# EN SIK GELDIGI ZAMANDAKI WIFI BILGILERINI CEK VE ONLARI CLUSTER'LARA AYIR(1-10)
# - Yeni dataframe'e her kolonu cekmeye calis
# - mongo yapısını duzelt

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
                #print("Feature[Date].minute Second: ",feature["Date"].minute,"          ",i) 
                #print("Feature[Date].second Second: ",feature["Date"].second,"           ",j)
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
resultDf.to_csv("urls_and_src_data.csv")
# resultDf.to_csv("flows_and_allow_data.csv")
