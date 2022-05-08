import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

datas = pd.read_excel("tenis.xlsx")
print(datas.isnull().sum()) # not missing value

# Kategorik dataların binarizasyonunu yapma(Label Encoder(), OneHotEncoder())
datas2 = datas.apply(preprocessing.LabelEncoder().fit_transform)
c = datas2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder() # datayı binary(0-1) olarak yazma 
c = ohe.fit_transform(c).toarray()
print(c)

#Kategorik dataları DataFrame'e çevirme
weatherForecast = pd.DataFrame(data = c,index=range(14), columns=['o','r','s'])
lastDatas = pd.concat([weatherForecast,datas.iloc[:,1:3]],axis=1)
lastDatas = pd.concat([datas2.iloc[:,-2:],lastDatas],axis=1)

# Dataları train-test olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(lastDatas.iloc[:,:-1],lastDatas.iloc[:,-1:],test_size=0.33,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

X = np.append(arr = np.ones((14,1)).astype(int), values = lastDatas.iloc[:,:-1],axis = 1)
X_l = lastDatas.iloc[:,:-1].values
X_l = np.array(X_l,dtype=(float))
model = sm.OLS(lastDatas.iloc[:,-1:],X_l).fit()
print(model.summary())  

#En Küçük Kareler(OLS): Regresyon sonuçları hakkında kapsamlı bir açıklama veren bir tablo elde etmek için kullanılır.
lastDatas = lastDatas.iloc[:,1:]
X = np.append(arr = np.ones((14,1)).astype(int), values = lastDatas.iloc[:,:-1],axis = 1)
X_l = lastDatas.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=(float))
model = sm.OLS(lastDatas.iloc[:,-1:],X_l).fit()
print(model.summary())
 
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
'''
outlook = df.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(df.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
print(outlook)

windy = df.iloc[:,3:4].values
le = preprocessing.LabelEncoder()
windy[:,0] = le.fit_transform(df.iloc[:,3:4])
ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()
print(windy)

play = df.iloc[:,4:5].values
le = preprocessing.LabelEncoder()
play[:,0] = le.fit_transform(df.iloc[:,4:5])
ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()
print(play)

outlookResult = pd.DataFrame(data = outlook, index = range(14), columns = ['overcast','rainy','sunny'])
windyResult = pd.DataFrame(data = windy[:,:1], index = range(14), columns = ['windy'])
playResult = pd.DataFrame(data = play[:,:1], index = range(14), columns = ['play'])
other = pd.DataFrame(data = df.iloc[:,1:3], index = range(14), columns = ['temperature','humidity'])

s = pd.concat([other,playResult,outlookResult],axis=1)
s2 = pd.concat([s,windyResult],axis=1)

x_train, x_test, y_train, y_test = train_test_split(s,windyResult,test_size=0.33, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


temperature = s2.iloc[:,:1].values
data = s2.iloc[:,1:]
x_train, x_test, y_train, y_test = train_test_split(data,temperature,test_size=0.33,random_state=0)
regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)

humidity = s2.iloc[:,1:2].values
left = s2.iloc[:,0]
right = s2.iloc[:,2:]
data = pd.concat([left,right],axis=1)
x_train, x_test, y_train, y_test = train_test_split(data,humidity,test_size=0.33,random_state=0)
regressor3 = LinearRegression()
regressor3.fit(x_train,y_train)

# Amaç oluşturduğumuz regresyon ile ilgili istatiksel değer oluşturmak
X = np.append(arr = np.ones((14,1)).astype(int), values = data, axis=1) # y = ax+b, b sabiti için datanın başına array olarak 22 tane bir ekliyoruz. # axis=1 bir kolon olarak eklemesi
X_l = data.iloc[:,]

X_l = data.iloc[:,[0,1,2,3,4,5]].values # 6 kolonun hepsini aldık
X_l = np.array(X_l,dtype=float) # bağımsız değişkenleri içeren dizi(ax+b)
# humidity => y=ax+b değişkenindeki bağımlı değişken,çıktı(y) değişkeni
# Buradaki amaç kolonların tek tek boy(humidity) üzerindeki etkisini ölçmek
model = sm.OLS(humidity, X_l).fit() # sm.OLS => istatiksel değerleri çıkartmaya yarıyor. 
print(model.summary()) # bu output'da en yüksek pi(x5) değerine sahip olanı eleyeceğiz.

X_l = data.iloc[:,[0,1,2,3,4]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(humidity, X_l).fit() # sm.OLS => istatiksel değerleri çıkartmaya yarıyor. 
print(model.summary()) 

data = data.iloc[:,[0,1,2,3,4]]
x_train, x_test, y_train, y_test = train_test_split(data,humidity,test_size=0.33,random_state=0)
regressor4 = LinearRegression()
regressor4.fit(x_train,y_train)
y_pred = regressor4.predict(x_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
'''




'''
x_train = x_train.sort_index(axis=1)  #################### Buraya sonra bak
y_train = y_train.sort(axis=1)
plt.plot(x_train,y_train,marker=".",markersize=10,markerfacecolor="red")
plt.plot(x_test,y_pred,marker="+",markersize=10)
plt.xlabel("Data")
plt.ylabel("Humidity")
plt.title("Hava Tahminleri")
plt.legend(["data","humidity"])
'''
# Amaç oluşturduğumuz regresyon ile ilgili istatiksel değer oluşturmak
#X = np.append(arr = np.ones((14,1)).astype(int), values = data, axis=1) # y = ax+b, b sabiti için datanın başına array olarak 22 tane bir ekliyoruz. # axis=1 bir kolon olarak eklemesi




