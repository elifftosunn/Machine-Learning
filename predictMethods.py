import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


datas = pd.read_excel("maaslarNew.xlsx")
x = datas.iloc[:,2:3].values
y = datas.iloc[:,5:].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),"b-*")
plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("LinearRegression")
plt.show()
print(lin_reg.predict([[6.5]]),"\n",lin_reg.predict([[20]]))

model = sm.OLS(lin_reg.predict(x),x)
print("Linear Regression R2 Value:",r2_score(y,lin_reg.predict(x)))
print("Linear Regression OLS:",model.fit().summary())


poly_reg = PolynomialFeatures(degree=8)
x_poly = poly_reg.fit_transform(x, y)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color="purple")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), "y-*")
plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("PolynomialFeatures")
plt.show()
#print(lin_reg.predict(poly_reg.fit_transform([[6.5]])),"\n",lin_reg.predict(poly_reg.fit_transform([[20]])))
model2 = sm.OLS(y,lin_reg2.predict(poly_reg.fit_transform(x)))
print("Polynomial Regression R2 Value:",r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))
print("Polynomial Regression OLS:",model2.fit().summary())


sc_x = StandardScaler()
x_scaler = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_scaler = sc_y.fit_transform(y)
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaler, y_scaler) 

plt.scatter(x_scaler, y_scaler, color="black")
plt.plot(x_scaler,svr_reg.predict(x_scaler), "g-*")
plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("Support Vector Regression (SVR)")
plt.show()
py_pred = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[6.5]]))))
py_pred2 = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[20]]))))
#py_pred3 = sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[5.5]]))))

print(py_pred,"\n",py_pred2)
model3 = sm.OLS(y,svr_reg.predict(x_scaler))
print("Support Vector Regression (SVR) R2 Value:",r2_score(y_scaler,svr_reg.predict(x_scaler)))  
print("Support Vector Regression OLS:",model3.fit().summary())

dt_reg = DecisionTreeRegressor()
dt_reg.fit(x, y)

plt.scatter(x, y, color="#B388A0")
plt.plot(x,dt_reg.predict(x), color="#70FFFF")
plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("Decision Tree Regression")
plt.show()

model4 = sm.OLS(y,dt_reg.predict(x))
print("Decision Tree Regression R2 Value:",r2_score(y,dt_reg.predict(x)))
print("Decision Tree Regression OLS:",model4.fit().summary())

rf_reg = RandomForestRegressor()
rf_reg.fit(x,y)

plt.scatter(x, y, color="#7051FF")
plt.plot(x,rf_reg.predict(x), color="#702400")
plt.xlabel("Unvan Seviyesi")
plt.ylabel("Maas")
plt.title("Random Forest Regression")
plt.show()

model5 = sm.OLS(y,rf_reg.predict(x))
print("Random Forest Regression R2 Value:",r2_score(y, rf_reg.predict(x))) 
print("Random Forest Regression OLS:",model5.fit().summary())


'''
3 parametreli olarak

linear
R-squared (uncentered):                   0.970
Polynomial Regression
R-squared (uncentered):                   0.858 (degree=4 olduğunda 1.000) 
Support Vector Regression (SVR)
R-squared (uncentered):                   0.124
Decision Tree Regression
R-squared (uncentered):                   1.000
Random Forest Regression
R-squared (uncentered):                   0.962

1 parametreli olarak
linear
R-squared (uncentered):                   0.943
Polynomial Regression
R-squared (uncentered):                   0.851
Support Vector Regression (SVR)
R-squared (uncentered):                   0.2554
Decision Tree Regression
R-squared (uncentered):                   0.851
Random Forest Regression
R-squared (uncentered):                   0.850

'''