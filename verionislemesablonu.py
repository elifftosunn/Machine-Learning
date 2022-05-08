#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#2.veri onisleme
#2.1.veri yukleme
datas = pd.read_excel('sales.xlsx')
#pd.read_csv("veriler.csv")
#test
print(datas)

months = datas[['Months']]
print(months)
sales = datas[["Sales"]]
print(sales)
months2 = datas.iloc[:,0:1].values
print(months2)
sales2 = datas.iloc[:,1:2].values
print(sales2)

#verilerin egitim ve test icin bolunmesi
x_train, x_test, y_train, y_test = train_test_split(months,sales,test_size=0.33,random_state=0)

lr = LinearRegression() # dataları tek bir doğru üzerinde gösteriyor
lr.fit(x_train, y_train)
predict = lr.predict(x_test) # y_test ile predict'i karşılaştır



x_train = x_train.sort_index() # index'e göre sıralandığından ikisinin indexlerini de sıralamamız gerek.
y_train = y_train.sort_index()
plt.plot(x_train,y_train,marker="o",markersize=10,markerfacecolor="red")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Aylara Göre Satış Miktarı")
plt.plot(x_test,predict,marker="+",markersize=10)
plt.legend(['data','model'])





'''
#verilerin olceklenmesi
sc = StandardScaler()
X_train = sc.fit_transform(x_train) #modeli oluşturmak için kullandığımız datalar
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(y_train) #modeli oluşturmak için kullandığımız datalar
Y_test = sc.fit_transform(y_test)

# model inşası (LinearRegression)
lr = LinearRegression()
lr.fit(X_train,Y_train) 

# X_train'den Y_train'i öğrendi modeli inşa ederken bu iki verileri kullandı burada da biz X_test'i veriyoruz Y_test'i tahmin ediyor.

predict = lr.predict(X_test)
'''

