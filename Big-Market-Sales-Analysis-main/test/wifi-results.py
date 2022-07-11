import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# RepaeatNumber => Kac kez o minute and second'de entry yapildigi
flowAllowWifi = pd.read_csv("onBinDataFlowsAllowWifi.csv")


print(flowAllowWifi.describe())
plt.figure(figsize=(20,5),dpi=300)
sns.countplot(flowAllowWifi["mac_or_request"])
plt.xticks(rotation=60,fontsize=14)
plt.title("Flow_Or_Allow  Mac_or_Request Count")
plt.show()

# print(len(flowAllowWifi.loc[flowAllowWifi["mac_or_request"] == "mac=F4:46:37:8A:F5:A7"]))
'''
in 10000 data more 950 freq (mac=F4:46:37:8A:F5:A7) mac address entry yapmis 
'''

plt.figure(figsize=(20,5),dpi=300)
sns.boxplot(data = flowAllowWifi, x = "mac_or_request", y = "RepeatNumber")
plt.xticks(rotation=60,fontsize=14)
plt.title("Flow_Or_Allow Wifi Repeat Number")
plt.show()

plt.figure(figsize=(20,5),dpi=300)
sns.barplot(data = flowAllowWifi, x = "mac_or_request", y = "Minute") # => more around 20 minute 
plt.xticks(rotation=60,fontsize=14)
plt.title("Flow_Or_Allow Wifi Minute by Mac_or_Request")
plt.show()

maxFlowAllow = flowAllowWifi.sort_values(["RepeatNumber"],ascending=False).head(42)
print("maxFlowAllow:protocol=tcp => ",len(maxFlowAllow.loc[maxFlowAllow["protocol"] == "protocol=tcp"])) # => 10 tcp, 32 udp used
# protocol => udp kullanildiginda daha fazla giris denemeleri oluyor
print(maxFlowAllow["mac_or_request"])
  
'''
mac=F4:D4:88:8A:A4:63
mac=F4:46:37:8B:43:EA
mac=F4:46:37:8B:66:DB
more repeat number in 10000 data, 34th minute and 50th second, 42 freq
'''
minFlowAllow = flowAllowWifi.sort_values("RepeatNumber",ascending=True).head(188)
print(minFlowAllow["mac_or_request"].value_counts())
'''
mac=BC:09:1B:F0:66:C1    31
mac=F4:46:37:8B:43:EA    28
mac=F4:46:37:8A:F5:A7    20
mac=F4:46:37:D8:6D:B9    20
mac=20:3C:AE:E3:37:D0    17
mac=F4:D4:88:8A:A4:63    14
mac=F4:46:37:8B:43:36    12
mac=BC:09:1B:F8:BA:51    12
mac=4C:02:20:07:7B:B3    10
mac=F4:46:37:8B:1F:28     7
mac=4E:45:D9:75:C4:C9     7
mac=BC:09:1B:DD:22:28     6
mac=F4:46:37:8B:66:DB     4
Name: mac_or_request, dtype: i

mac=BC:09:1B:F0:66:C1 => in 10000 data different times 31 freq entry(arada seconds(1-2 second) olan da var)
mac=F4:46:37:8B:43:EA => in 10000 data different times 28 freq entry
'''

print("flowAllowWifi least repeat number =>  ",len(flowAllowWifi.loc[flowAllowWifi["RepeatNumber"] == 1]))

print("flowAllowWifi more repeat number =>  ",len(flowAllowWifi.loc[flowAllowWifi["RepeatNumber"] == 42]))

moreMac = minFlowAllow.loc[minFlowAllow["mac_or_request"] == "mac=BC:09:1B:F0:66:C1"]

print("minFlowAllow:protocol=tcp => ",len(minFlowAllow.loc[minFlowAllow["protocol"] == "protocol=tcp"])) # => 104 tcp, 84 udp used
# protocol => tcp kullanildiginda daha az giris denemeleri oluyor.



urlsSrcWifi = pd.read_csv("onBinDataUrls_Src.csv")

plt.figure(figsize=(20,5),dpi=300)
sns.countplot(urlsSrcWifi["mac_or_dst"])
plt.xticks(rotation=60,fontsize=14)
plt.title("urlsSrcWifi Mac_or_Dst Count")
plt.show()

plt.figure(figsize=(20,5),dpi=300)
sns.barplot(data = urlsSrcWifi, x = "mac_or_dst", y = "RepeatNumber")
plt.xticks(rotation=60,fontsize=14)
plt.title("urlsSrcWifi Mac_or_Dst Repeat Number")
plt.show()
'''
More Repeat Number 
'''

plt.figure(figsize=(20,5),dpi=300)
sns.boxplot(data = urlsSrcWifi, x = "mac_or_dst", y = "RepeatNumber")
plt.xticks(rotation=60,fontsize=14)
plt.title("urlsSrcWifi Mac_or_Dst Repeat Number Outlier Values")
plt.show()

maxUrlOrSrc = urlsSrcWifi.sort_values("RepeatNumber",ascending=False)
'''
More with 14(same second and same minute) repeat 27th minute 14th second and 39th minute 50th second'de entry yapilmis

'''

maxUrlsWifi = urlsSrcWifi.loc[urlsSrcWifi["RepeatNumber"] == 14]
print(maxUrlsWifi["mac_or_dst"].value_counts())

'''

mac=4C:02:20:07:7B:B3    13
mac=F4:46:37:8B:1F:28    12
mac=BC:09:1B:F8:BA:51     1
mac=20:3C:AE:E3:37:D0     1
mac=F4:46:37:8B:43:EA     1

This mac address'leri  14 freq(same second and same minute) 39th minute 50th second and 27th minute 14th second  giris yapilmis.
More freq  mac=4C:02:20:07:7B:B3 address in all data 167 freq var and 
'''

print(urlsSrcWifi["mac_or_dst"].value_counts())

print(urlsSrcWifi["mac_or_dst"].value_counts())


data = urlsSrcWifi.loc[urlsSrcWifi["mac_or_dst"] == "mac=BC:09:1B:DD:22:28"]
print(data["RepeatNumber"].sort_values(ascending=False))








