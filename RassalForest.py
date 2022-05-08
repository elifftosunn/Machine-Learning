import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

datas = pd.read_excel("maaslar.xlsx")
x = datas.iloc[:,1:2].values
y = datas.iloc[:,2].values


sc_x = StandardScaler()
x_scaler = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_scaler = sc_y.fit_transform(y.reshape(-1,1))

svr_reg = SVR(kernel=("rbf"))
svr_reg.fit(x_scaler, y_scaler)
plt.scatter(x_scaler, y_scaler, color="red")
plt.plot(x_scaler,svr_reg.predict(x_scaler),color="blue")
plt.show()
#py_pred3 = sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[18]]))))
py_pred = sc_y.inverse_transform(svr_reg.predict(sc_x.transform(np.array([[2.5]]))))
print(py_pred)
# DecisionTree: bildiğimiz verileri tahmin ediyor, bilmediği verilere geçtiğinde bildiği verilerle aynı sonucu döndürüyor.
# DECISION TREE REGRESSION
r_dt = DecisionTreeRegressor(random_state=(0))
r_dt.fit(x,y)
Z = x + 0.5
K = x - 0.4
plt.scatter(x, y, color="black")
plt.plot(x,r_dt.predict(x), color="grey")

plt.plot(x,r_dt.predict(Z), color="green")
plt.plot(x,r_dt.predict(K), color="yellow")
plt.show()

print(r_dt.predict([[11]]),"\n",r_dt.predict([[6.6]]))

# RANDOM TREE REGRESSION
rf_reg = RandomForestRegressor(random_state=0, n_estimators = 10) # n_estimators: kaç tane desicion tree çizileceği
rf_reg.fit(x, y.ravel()) # birden fazla decision tree'lerin average'ını döndürüyor.
print(rf_reg.predict([[6.6]]))
plt.scatter(x, y, color="blue")
plt.plot(x,rf_reg.predict(x),"r-*")
plt.plot(x,rf_reg.predict(Z), color="green")
plt.plot(x,rf_reg.predict(K), color="grey")
plt.show()  




