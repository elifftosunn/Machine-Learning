import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


datas = pd.read_excel("maaslar.xlsx")
levelOfEducation = datas.iloc[:,1:2]
salary = datas.iloc[:,2:]
x = datas.iloc[:,1:2].values # aslında lineer olmadığını bildiğim veriler üzerinde bir lineer model oluşturmak
y = datas.iloc[:,2:].values # önce dataFrame slicing daha sonra (.values) ile numpy array dönüşümü

# linear regression(doğrusal model oluşturma)
lin_reg = LinearRegression()
lin_reg.fit(x,y)
y_pred = lin_reg.predict(x)
#datayı görselleştirme
plt.scatter(x, y,color="red")
plt.plot(x, y_pred,color="blue")
plt.show()

# polynomial regression
poly_reg = PolynomialFeatures(degree=2) #2.dereceden bir polinom objesi oluştur.
x_poly = poly_reg.fit_transform(x) # x(x^0,x^1,x^2) 1'den 10'a kadar giden sayılar olduğundan kareleri de 1'den 10'a kadar giden sayılardır.
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) # x_poly'yi(1,2,4 - 1,3,9 - 1,10,100) kullanarak y=ax^2+bx+c a,b,c değerlerini öğren
#datayı görselleştirme
plt.scatter(x, y,color="red")
plt.plot(x,lin_reg2.predict(x_poly),color="blue") # linear regression sonucunu çizceğin her bir data point(x) için önce polynomal dönüşümünü(ax^2+bx+c) yap ve ondan sonra çiz
plt.show()


poly_reg = PolynomialFeatures(degree=4) #4.dereceden bir polinom objesi oluştur.
x_poly = poly_reg.fit_transform(x) # x(x^0,x^1,x^2) 1'den 10'a kadar giden sayılar olduğundan kareleri de 1'den 10'a kadar giden sayılardır.
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y) # x_poly'yi kullanarak y=ax^2+bx+c a,b,c değerlerini öğren
plt.scatter(x, y, color="red")
plt.plot(x,lin_reg2.predict(x_poly),color="blue") # linear regression sonucunu çizceğin her bir data point(x) için önce polynomal dönüşümünü(ax^2+bx+c) yap ve ondan sonra çiz
plt.show()



#predicts
print(lin_reg.predict([[11]]))#direk 11'i tahmin et demek doğrusal(linear) bir tahmin yap demek
print(lin_reg.predict([[6.6]]))
# Burada ise polynomal olarak dönüşümünü sağlıyoruz ve tahmin gerçek değerlere daha fazla yaklaşıyor.
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


#veri ön işleme
sc1 = StandardScaler()
x_scaler = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scaler = sc2.fit_transform(y)

svr_reg = SVR(kernel="rbf") #kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
svr_reg.fit(x_scaler,y_scaler) # modeli eğitiyoruz
plt.scatter(x_scaler, y_scaler, color="red")
plt.plot(x_scaler, svr_reg.predict(x_scaler), "b-*")

py_pred2 = svr_reg.predict(np.array([11]).reshape(-1,1))
py_pred = svr_reg.predict(np.array([6.5]).reshape(-1,1))
print(py_pred,"\n",py_pred2)
py_pred3 = sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[5.5]]))))
print(py_pred3) # inverse_transform(). Bu metod ölçeklendirilmiş değerleri alıp gerçek  değerlere geri dönüştürüyor.
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
y_pred = lin_reg.predict(x_test)

plt.plot(x_train,y_train,"r-*")
plt.plot(x_test,y_pred,"b-+",markersize=12)

'''









