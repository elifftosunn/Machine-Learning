import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer # Sci - Kit Learn(Eksik verilerin işlenmesi)
from sklearn import preprocessing # kategorik dataların dönüşümü için
from sklearn.model_selection import train_test_split # datayı train ve test olarak ikiye bölme
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # modelin ve modeldeki değişkenlerin başarısı ile ilgili 

datas = pd.read_excel("datas.xlsx")
print(datas[['age','weight']])

# eksik datas
# sci - kit learn
imputer = SimpleImputer(missing_values=np.nan,strategy="mean") #NaN olan değerler için diğer sayıların ortalamasını alıp NaN değerlerin yerine yazma
Age = datas.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4]) # fit ile eğitme
Age[:,1:4] = imputer.transform(Age[:,1:4]) # transform ile veriyi dönüştürüyor ve verimli hale getiriyor
print(Age)


# (Encoder) kategorik datas => Numerik
country = datas.iloc[:,0:1].values
print(country)
le = preprocessing.LabelEncoder() #LabelEncoder: Kategorik verilere sayısal bir değer atar.(0-1-2-3)
country[:,0] = le.fit_transform(datas.iloc[:,0])
print(country)
ohe = preprocessing.OneHotEncoder() #OneHotEncoder: kategorik verilerin binarizasyonunu(0-1) gerçekleştirmemizi sağlar.
country = ohe.fit_transform(country).toarray() # numpy array
print(country)


g = datas.iloc[:,-1:].values
print(g)
le = preprocessing.LabelEncoder()
g[:,-1] = le.fit_transform(datas.iloc[:,-1])
print(g)
ohe = preprocessing.OneHotEncoder()
g = ohe.fit_transform(g).toarray()
print(g) # erkek: 1 , bayan: 0


# dataları dataFrame'e dönüştürme
countryResult = pd.DataFrame(data=country, index = range(22), columns = ["fr","tr","us"])
print(countryResult)
ageResult = pd.DataFrame(data=Age, index = range(22), columns = ['Height','Weight','Age'])
print(ageResult) 

#gender = datas.iloc[:,-1].values # -1 = sondan bir kolon demektir
#print(gender)
genderResult = pd.DataFrame(data = g[:,:1], index = range(22), columns=['gender'])
print(genderResult)


# farklı farklı dataFrame'leri alıp tek bir dataframe'de merge,concat etmek
s = pd.concat([countryResult,ageResult],axis=1) # satır bazlı ayırıyor
print(s)
s2 = pd.concat([s,genderResult],axis=1) # x = height,weight,age,country , y = gender
print(s2)


# Veriyi önce dikey eksende bağımlı ve bağımsız değişkenler olarak ikiye ayırıyoruz, sonra da yatay eksende train ve test olarak ikiye ayırıyoruz..
x_train, x_test, y_train, y_test = train_test_split(s,genderResult,test_size=0.33,random_state=0)
#print(x_train.shape,"\n",x_test.shape,"\n",y_train.shape,"\n",y_test.shape)

regressor = LinearRegression()
regressor.fit(x_train,y_train) # x_train'den y_train'i öğren diyoruz

y_pred = regressor.predict(x_test) # x'in test olarak ayrılmış kısmını tahmin et ve y_pred'e yaz. Amaç aralarında lineer bir model kurmak(machine learning)

height = s2.iloc[:,3:4].values
left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

data = pd.concat([left,right],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,height,test_size=0.33,random_state=0) # x=data,y=height

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train) # x_train'i y_train ile eğit

y_pred = regressor2.predict(x_test) # x_test'den y_test'i tahmin et
# y_test ve y_pred'i karşılaştır
# acaba doğru columns'ları mı kullanıyoruz, kolonların hepsini almakla hata mı yapıyoruz?



# Amaç oluşturduğumuz regresyon ile ilgili istatiksel değer oluşturmak
X = np.append(arr = np.ones((22,1)).astype(int), values = data, axis=1) # y = ax+b, b sabiti için datanın başına array olarak 22 tane bir ekliyoruz. # axis=1 bir kolon olarak eklemesi

X_l = data.iloc[:,[0,1,2,3,4,5]].values # 6 kolonun hepsini aldık
X_l = np.array(X_l,dtype=float) # bağımsız değişkenleri içeren dizi(ax+b)
# height => y=ax+b değişkenindeki bağımlı değişken,çıktı(y) değişkeni
# Buradaki amaç kolonların tek tek boy(height) üzerindeki etkisini ölçmek
model = sm.OLS(height, X_l).fit() # sm.OLS => istatiksel değerleri çıkartmaya yarıyor. 
print(model.summary()) # bu output'da en yüksek pi(x5) değerine sahip olanı eleyeceğiz.

X_l = data.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())

X_l = data.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())

# İstatiksel değerlere göre yeni regresyon modeli kurma
data = data.iloc[:,[0,1,2,3]]
x_train, x_test, y_train, y_test = train_test_split(data,height,test_size=0.33,random_state=0)
regressor3 = LinearRegression()
regressor3.fit(x_train,y_train)

y_pred = regressor3.predict(x_test)

'''
height = s2.iloc[:,3:4].values
left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

data = pd.concat([left,right],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,height,test_size=0.33,random_state=0) # x=data,y=height

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train) # x_train'i y_train ile eğit

y_pred = regressor2.predict(x_test) # x_test'den y_test'i tahmin et
'''





















