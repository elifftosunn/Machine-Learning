from helpers.mongoStructure import *

df = pd.read_csv("datas/processedDatas/mongo-wifi.csv")
print(df.mac_or_request.value_counts())
# print(df["Server"].value_counts())
# print(df["allow_or_src"].value_counts())
#print(df["mac_or_request"].value_counts())
#print(df["dport"].value_counts())

print(mongoResult._mongoInformation.databaseNames())
#print(mongoResult._mongoInformation.collectionNames("wifiData"))
# for value in mongoResult._mongoInformation.dataFindAll():
#     print(value)

# for value in mongoResult._mongoInformation.dataFindQuery({"Server":"server-176.53.2.142.as42926.net"}):
#     print(value)
# print(mongoResult._mongoInformation.dataFindOne({"Server":"server-176.53.2.142.as42926.net"}))
# print(mongoResult._mongoInformation.dataFindOne({"_id":ObjectId("62be8c87b56e0328f9229a48")}))
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


# HANGI SERVER'LAR DAHA HIZLI VE HANGI SERVER'DAN KAC ADET DST GELIYOR?

def dstNumberOfServer(columns,queryServerDst,dst):
    FeatureResult = mongoResult._mongoInformation.dataFindQuery(queryServerDst,columns)
    countDst = 0
    for value in FeatureResult:
        result = value["mac_or_dst"] == dst
        if result == True:
            countDst += 1
    return countDst   
def numberServer(columns,queryServer,server):
    FeatureResult = mongoResult._mongoInformation.dataFindQuery(queryServer,columns)
    countServer = 0
    for value in FeatureResult:
        result = value["Server"] == server
        if result == True:
            countServer += 1
    return countServer
def NumberOfServerResult(): 
    queryServerDst =  {"mac_or_dst":{"$regex":"^dst="}}
    columns = {"_id":0,"mac_or_dst":1,"Server":2,"Date":3}
    servers = ["server-176.53.2.142.as42926.net",
               "195.175.205.46.static.turktelekom.com.tr",
               "merakimulti"]
    resultDst = mongoResult._mongoInformation.dataFindQuery(queryServerDst,columns)
    valueDstTotal, eachDstNumber, serverNumber, serverTotal = [],{},{},[]
    countResultServerDstMinute, countResultServerDstSecond = {},{}
    for value in resultDst:
        if value["mac_or_dst"] not in valueDstTotal:
            valueDstTotal.append(value["mac_or_dst"])
    for dst in valueDstTotal: # eachDstNumber'a eklenen her degerin unique olması gerekkkkk!!!!!!!!!!!!!
        countDst = dstNumberOfServer(columns, queryServerDst, dst) # number for each dst
        eachDstNumber[dst] = countDst # key => dst=20.50.80.209  value => count:  28
    for value in resultDst:
        print(value,"\n") # BURADA SORUN VAR SERVER CIKMIYOR AMA MONGODB'DE SORGULANDIGINDA TUM OZELLIKLER CIKIYOR SADECE
        # MAC_OR_DST DST ILE BASLAYANLAR SINIRLANIYORRRRRRRRRRRRRRRRRRRR
        # if value["Server"] not in serverTotal:
        #     serverTotal.append(value["Server"])

    # for server in serverTotal:
    #     countServer = numberServer(columns,queryServerDst,server)
    #     serverNumber[server] = countServer 
    # for key,value in serverNumber.items():
    #     print(key,":",value)
    #     #{"$and":[{"mac_or_dst":"dst=172.18.14.96"},{"protocol":"protocol=udp"}]}
        
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns, queryServerDst, i, i) 
        countResultServerDstMinute[i] = countMinute 
        countResultServerDstSecond[i] = countSecond  
    # for key,value in countResultServerDstMinute.items():
    #     print("Server Dst Minute ",key,":",value)
    # for key,value in countResultServerDstSecond.items():
    #     print("Server Dst Second ",key,":",value)
    #return countResultServerDstMinute,countResultServerDstSecond, eachDstNumber, serverNumber
        
#NumberOfServerResult()            

# kac tane disaridan dst geliyor? kac saniyede geliyor ve en cok kacıncı saniyede geliyor?
def dstResult():
    columns = {"_id":0,"mac_or_dst":1,"Date":2,"protocol":3}
    queryDst = {"mac_or_dst":{"$regex":"^dst"}}
    countResultDstSecond, countResultDstMinute = {}, {}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns,queryDst,i,i)
        countResultDstSecond[i] = countSecond
        countResultDstMinute[i] = countMinute
    # for key,value in countResultDstMinute.items():
    #     print("Dst Minute",key,":",value) # Max Freq: 316
    # for key,value in countResultDstSecond.items():
    #     print("Dst Second",key,":",value) # Max Freq: 219
    return countResultDstMinute,countResultDstSecond
# ************************** MAC *****************************************
def macNumber(queryMac, columns, mac):
    FeatureResult = mongoResult._mongoInformation.dataFindQuery(queryMac,columns)
    countMac = 0
    for value in FeatureResult:
        result = value["mac_or_request"] == mac
        if result == True:
            countMac += 1
    return countMac   
def macOrRequestResult():
    columns = {"_id":0,"mac_or_request":1,"Date":2}
    queryMac = {"mac_or_request":{"$regex":"^mac"}}
    valueMacOrRequest = []
    countResultMacSecond, countResultMacMinute, macNum = {},{},{}
    FeatureResult = mongoResult._mongoInformation.dataFindQuery(queryMac,columns)
    for value in FeatureResult:
        if value["mac_or_request"] not in valueMacOrRequest:
            valueMacOrRequest.append(value["mac_or_request"])
    for mac in valueMacOrRequest:
        countMac = macNumber(queryMac, columns, mac)
        macNum[mac] = countMac
    # for key,value in macNum.items():
    #     print(key,"count: ",value)   
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns, queryMac, i, i)
        countResultMacSecond[i] = countSecond 
        countResultMacMinute[i] = countMinute
    # for key,value in countResultMacSecond.items():
    #     print(key,"second: ",value)
    # for key,value in countResultMacMinute.items():
    #     print(key,"minute: ",value)
    return countResultMacMinute,countResultMacSecond, macNum
#macOrRequestResult()

# - kablosuz agdan hedefe kac saniyede gidiliyor(src-dst-second)

'''
allow                     7821
deny                        55
src=172.19.0.172:63637       9
-c                           8
type=dfs_event               4
src=172.19.0.172:63508       1
src=172.19.0.179:50655       1
src=172.19.0.166:65044       1
src=172.19.0.71:55705        1
src=172.19.0.173:54845       1
'''

# HANGI PROTOCOL ILE GELIYOR VE PROTOCOL'UN GELME SURELERI NEDIR?
df = pd.read_csv("datas/processedDatas/mongo-wifi.csv")
# print(df["Server"].value_counts())
# print(df["allow_or_src"].value_counts())
#print(df["mac_or_request"].value_counts())
print(df["TR_IST_AP"].value_counts())
'''
server-176.53.2.142.as42926.net             9011
195.175.205.46.static.turktelekom.com.tr     978
merakimulti                                   11
'''
def protocolUdp():
    queryProtocolUdp = {"protocol":"protocol=udp"}
    columns = {"_id":0,"protocol":1,"Date":2}
    countResultUdpSecond, countResultUdpMinute = {}, {}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns,queryProtocolUdp,i,i)
        countResultUdpSecond[i] = countSecond 
        countResultUdpMinute[i] = countMinute 
    # for key,value in countResultDstMinute.items():
    #     print("Protocol Udp Minute",key,":",value) # Max Value Minute: 19 : 316
    # for key,value in countResultDstSecond.items():
    #     print("Protocol Udp Second",key,":",value) # Max Value Second: 50 : 219
    return countResultUdpMinute,countResultUdpSecondSS
#protocolUdp()
def protocolTcp():
    queryProtocolTcp = {"protocol":"protocol=tcp"}
    columns = {"_id":0,"protocol":1,"Date":2}
    countResultTcpSecond, countResultTcpMinute = {}, {}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns,queryProtocolTcp,i,i)
        countResultTcpSecond[i] = countSecond 
        countResultTcpMinute[i] = countMinute 
    # for key,value in countResultDstMinute.items():
    #     print("Protocol Tcp Minute",key,":",value) # Max Value Minute: 19 : 316
    # for key,value in countResultDstSecond.items():
    #     print("Protocol Tcp Second",key,":",value) # Max Value Second: 56 : 174
    return countResultTcpMinute,countResultTcpSecond
#protocolTcp()
def flows():
    queryFlows = {"flow_or_url":"flows"}
    columns = {"_id":0,"flow_or_url":1,"Date":2}
    countResultFlowsSecond, countResultFlowsMinute = {}, {}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns, queryFlows, i, i)
        countResultFlowsSecond[i] = countSecond 
        countResultFlowsMinute[i] = countMinute 
    # for key, value in countResultFlowsSecond.items():
    #     print("Flows Second ",key,":",value) # Flows Second  50 : 219
    # for key, value in countResultFlowsMinute.items():
    #     print("Flows Minute ",key,":",value) # Flows Minute  19 : 316
    return countResultFlowsMinute,countResultFlowsSecond
#flows()
def urls():
    queryUrls = {"flow_or_url":"urls"}
    columns = {"_id":0,"flow_or_url":1,"Date":2}
    countResultUrlSecond, countResultUrlMinute = {},{}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns, queryUrls, i, i)
        countResultUrlMinute[i] = countMinute 
        countResultUrlSecond[i] = countSecond 
    # for key,value in countResultUrlMinute.items():
    #     print("Url Minute",key,":",value) # Url Minute 22 : 117
    # for key,value in countResultUrlSecond.items():
    #     print("Url Second",key,":",value) # Url Second 50 : 81
    return countResultUrlMinute, countResultUrlSecond    
#urls()
def dport():
    queryDport = {"dport":{"$regex":"^dport"}} # 7776 data
    columns = {"_id":0,"dport":1,"Date":2}
    splitData = []
    result= mongoResult._mongoInformation.dataFindQuery(queryDport,columns)
    for value in result:
        value["dport"] = str(value["dport"])
        splitDport = value["dport"].split("=")
        splitData.append(splitDport[1])  # the number of each dport address(443,53 vb.) => kac farklı dport address'i var gibi..
    countResultDportMinute, countResultDportSecond = {},{}
    for i in range(60):
        countSecond, countMinute = SecondOrMinute(columns, queryDport, i, i)
        countResultDportMinute[i] = countMinute 
        countResultDportSecond[i] = countSecond 
    # for key,value in countResultDportMinute.items(): # Dport Minute 19 : 315
    #     print("Dport Minute",key,":",value) 
    # for key,value in countResultDportSecond.items(): # Dport Second 50 : 214
    #     print("Dport Second",key,":",value) 
    return countResultDportMinute, countResultDportSecond
#dport()


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
    max_value_dst,max_key_dst = int(),int()        
    for key,value in results[2].items():
        if value > max_value_dst:
            max_value_dst = value
            max_key_dst = key 
    # max_value_server,max_key_server = int(),int()        
    # for key,value in results[3].items():
    #     if value > max_value_server:
    #         max_value_server = value
    #         max_key_server = key 
    
    # for i in str(results).split(","): 
    #     max_value_server,max_key_server = int(),int()
    #     for key,value in results[i].items():
    #         if value > max_value_server:
    #             max_value_server = value
    #             max_key_server = key 
    print(max_key_minute,"Max Value Minute:",max_value_minute)
    print(max_key_second,"Max Value Second:",max_value_second)  
    print(max_key_dst,"Max Value Dst:",max_value_dst) 
    print(max_key_server,"Max Value Server:",max_value_server)
countResultDstMinute,countResultDstSecond = dstResult() # Max Value Minute: 19 : 316, Max Value Second: 50 : 219
countResultUdpMinute,countResultUdpSecond = protocolUdp() # Max Value Minute: 19 : 316, Max Value Minute: 50 : 219
countResultTcpMinute,countResultTcpSecond = protocolTcp() # Max Value Minute: 19 : 316, Max Value Second: 56 : 174
countResultFlowsMinute,countResultFlowsSecond = flows() # Flows Minute  19 : 316, Flows Second  50 : 219
countResultUrlMinute, countResultUrlSecond = urls() # Url Minute 22 : 117, Url Second 50 : 81
countResultDportMinute, countResultDportSecond = dport() # Dport Minute 19 : 315, Dport Second 50 : 214

#countResultServerDstMinute,countResultServerDstSecond, eachDstNumber, serverNumber = NumberOfServerResult()
countResultMacMinute,countResultMacSecond, macNum = macOrRequestResult() # SUNAAAA TEKRAR BAKKKKKKKKKK
#Results(countResultMacMinute,countResultMacSecond, macNum) 


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






'''
_id                      62be8c87b56e0328f922beb1
Server                   "merakimulti"
TR_IST_AP                 "CMD"
flow_or_url             "(/usr/bin/php5"
allow_or_src              "-c"
SNAT_or_DNAT              "/etc/php5/apache/php.ini"
mac_or_dst              "-f"  => -x or -f
mac_or_request        "/var/www/includes/check_sms.php)"
Date                2022-06-26T21:40:01.000+00:00
'''