import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
# SVR, verilerde doğrusal olmayanlığın varlığını kabul eder ve yetkin bir tahmin modeli sağlar.



datas = pd.read_excel("exampleData.xlsx")
x = datas.iloc[:,:1].values
y = datas.iloc[:,1].values
lin_reg = LinearRegression()
lin_reg.fit(x,y) # fit ile makineyi eğittik
y_pred = lin_reg.predict(x)

plt.scatter(x, y, color="red")
plt.plot(x,y_pred,"b-*") # linear(doğrusal) bir doğru
plt.show()

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(x, y ,color="blue")
plt.plot(x, lin_reg2.predict(x_poly), "r-*")
plt.show()

poly_reg2 = PolynomialFeatures(degree=8)
x_poly2 = poly_reg2.fit_transform(x) #modeli veri ile eğitiyor ve verimli hale getiriyor
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly2,y)
plt.scatter(x, y, color="green")
plt.plot(x,lin_reg3.predict(x_poly2), "r-*")
plt.show()     

# SVR, verilerde doğrusal olmayanlığın varlığını kabul eder ve yetkin bir tahmin modeli sağlar.
sc_x = StandardScaler()
x_scaler = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_scaler = sc_y.fit_transform(y.reshape(-1,1))
svr_reg = SVR(kernel=("rbf"))
svr_reg.fit(x_scaler,y_scaler)
plt.scatter(x_scaler, y_scaler, color="red")
plt.plot(x_scaler,svr_reg.predict(x_scaler),"b-*")
plt.show()

py_pred = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[12.5]]))))
print(py_pred) # inverse_transform(). Bu metod ölçeklendirilmiş değerleri alıp gerçek  değerlere geri dönüştürüyor.
#py_pred3 = sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[18]]))))

'''
datas = pd.read_excel('sales.xlsx')
x = datas.iloc[:,:1].values
y = datas.iloc[:,1].values

lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)
plt.scatter(x, y, color="red")
plt.plot(x,y_pred,"b-*")
plt.show()


poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(x, y, color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),"g-*")
plt.show() 


poly_reg = PolynomialFeatures(degree=8) #4.dereceden bir polinom objesi oluştur.
x_poly = poly_reg.fit_transform(x) # x(x^0,x^1,x^2) 1'den 10'a kadar giden sayılar olduğundan kareleri de 1'den 10'a kadar giden sayılardır.
print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y) # x_poly'yi kullanarak y=ax^2+bx+c a,b,c değerlerini öğren
plt.scatter(x, y, color="red")
plt.plot(x,lin_reg3.predict(x_poly),color="blue") # linear regression sonucunu çizceğin her bir data point(x) için önce polynomal dönüşümünü(ax^2+bx+c) yap ve ondan sonra çiz
plt.show()

#veri ön işleme (Feature Scaling)
sc1 = StandardScaler()
x_scaler = sc1.fit_transform(x)
sc2 = StandardScaler()# Gördüğümüz gibi feature scaling uyguladığımız yeni değişkenlerimiz yukarıda görülüyor. Bu işlemi kabaca km ile mm’yi işleme sokmadan önce ikisini de metreye çevirmek olarak düşünebiliriz.
y_scaler = sc2.fit_transform(y.reshape(-1,1))

# kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
# gamma{'scale', 'auto'} or float, default='scale'  
svr_reg = SVR(kernel="rbf") #kernel parametresinin karşılığı ‘rbf’ zaten varsayılan çekirdek. Bu şu demek: değişkenler arasındaki ilişki durumuna göre bir çekirdek seçilmeli
svr_reg.fit(x_scaler,y_scaler) #fit ile Makinemizi eğitelim.
plt.scatter(x_scaler, y_scaler, color="red")
plt.plot(x_scaler, svr_reg.predict(x_scaler), "b-*")

py_pred2 = svr_reg.predict(np.array([11]).reshape(-1,1))
py_pred = svr_reg.predict(np.array([6.5]).reshape(-1,1))
print(py_pred,"\n",py_pred2)
py_pred3 = sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[18]]))))
print(py_pred3) # inverse_transform(). Bu metod ölçeklendirilmiş değerleri alıp gerçek  değerlere geri dönüştürüyor.
'''



