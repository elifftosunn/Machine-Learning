import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer # Sci - Kit Learn(Eksik verilerin işlenmesi)
from sklearn import preprocessing # kategorik dataların dönüşümü için
from sklearn.model_selection import train_test_split # datayı train ve test olarak ikiye bölme
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix


datas = pd.read_excel("missingDatas.xlsx")
print(datas[['age','weight']])

# eksik datas
# sci - kit learn
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
Age = datas.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4]) # fit ile öğretip
Age[:,1:4] = imputer.transform(Age[:,1:4]) # transform ile öğrenmesini sağlıyoruz.
print(Age)


# (Encoder) kategorik datas => Numerik
country = datas.iloc[:,0:1].values
print(country)
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
print(country)
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray() # numpy array
print(country)

# dataları dataFrame'e dönüştürme
countryResult = pd.DataFrame(data=country, index = range(22), columns = ["fr","tr","us"])
print(countryResult)
ageResult = pd.DataFrame(data=Age, index = range(22), columns = ['Height','Weight','Age'])
print(ageResult) 

gender = datas.iloc[:,-1].values # -1 = sondan bir kolon demektir
print(gender)
genderResult = pd.DataFrame(data = gender, index = range(22), columns=['gender'])
print(genderResult)

# farklı farklı dataFrame'leri alıp tek bir dataframe'de merge,concat etmek
s = pd.concat([countryResult,ageResult],axis=1) # satır bazlı ayırıyor
print(s)
s2 = pd.concat([s,genderResult],axis=1)
print(s2)

# Veriyi önce dikey eksende bağımlı ve bağımsız değişkenler olarak ikiye ayırıyoruz, sonra da yatay eksende train ve test olarak ikiye ayırıyoruz..
x_train, x_test, y_train, y_test = train_test_split(s,genderResult,test_size=0.33,random_state=0)
print(x_train.shape,"\n",x_test.shape,"\n",y_train.shape,"\n",y_test.shape)

#(Öznitelik Ölçekleme) Train ve test datas birbirlerini göre ölçelendirildi.
#sc = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)















