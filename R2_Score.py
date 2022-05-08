import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

datas = pd.read_excel("maaslar.xlsx")
x = datas.iloc[:,1:2].values
y = datas.iloc[:,2].values


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
plt.scatter(x, y, color = "green")
plt.plot(x,lin_reg.predict(x), "r-*")
plt.title("Linear Regression")
plt.show()
print("Linear Regression R2 Values:",r2_score(y,lin_reg.predict(x)))


# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x, y)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
plt.scatter(x, y, color="red")
plt.plot(x,lin_reg.predict(poly_reg.fit_transform(x)),"b-*")
plt.title("Polynomial Regression")
plt.show()
print("Polnomial Regression R2 Values:",r2_score(y,lin_reg.predict(poly_reg.fit_transform(x))))



# Dataların Ölçeklenmesi
# Support Vector Regression (SVR)
sc_x = StandardScaler()
x_scaler = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_scaler = np.ravel(sc_y.fit_transform(y.reshape(-1,1)))

svr_reg = SVR(kernel=("rbf"))
svr_reg.fit(x_scaler,y_scaler)

plt.scatter(x_scaler,y_scaler,color="red")
plt.plot(x_scaler,svr_reg.predict(x_scaler), color="blue")
plt.title("Support Vector Regression")
plt.show()
#print(svr_reg.predict([[11]]),"\n",svr_reg.predict([[6.5]]))

print("Support Vector R2 Degeri:",r2_score(y_scaler,svr_reg.predict(x_scaler)))



# Decision Tree Regression
r_dt = DecisionTreeRegressor()
r_dt.fit(x,y)
K = x + 0.5
Z = x - 0.4
plt.figure(figsize=(12,7))
plt.scatter(x, y, color = "green")
plt.plot(x,r_dt.predict(x) , color="red")
plt.plot(x,r_dt.predict(K), color="blue")
plt.plot(x,r_dt.predict(Z), color="grey")
plt.title("Decision Tree Regression")
plt.show()


print("Decision Tree R2 Degeri:",r2_score(y,r_dt.predict(x))) # R2 değerine baktığımızda Decision Tree en iyi yöntemmiş gibi görünüyor ama belirli aralıklarda aynı değerleri verdiğinden iyi bir yöntem değil.
#print("Decision Tree R2 Degeri:",r2_score(y,r_dt.predict(K)))
#print("Decision Tree R2 Degeri:",r2_score(y,r_dt.predict(Z)))


# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0) # n_estimators: kaç tane desicion tree çizileceği
rf_reg.fit(x,y)
plt.scatter(x, y, color="green")
plt.plot(x,rf_reg.predict(x), color="grey")
plt.plot(x,rf_reg.predict(K), color="red")
plt.plot(x,rf_reg.predict(Z), color="blue")
plt.title("Random Forest Regression")
plt.show()



print("Random Forest R2 Degeri: ",r2_score(y,rf_reg.predict(x)))  # 1'e yaklaştıkça olumlu demek oluyor tahmin açısından
print("Random Forest R2 Degeri: ",r2_score(y,rf_reg.predict(K)))  # 1'e yaklaştıkça olumlu demek oluyor tahmin açısından
print("Random Forest R2 Degeri: ",r2_score(y,rf_reg.predict(Z)))  # 1'e yaklaştıkça olumlu demek oluyor tahmin açısından






