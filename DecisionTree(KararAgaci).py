import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

datas = pd.read_excel("maaslar.xlsx")
x = datas.iloc[:,1:2].values
y = datas.iloc[:,2].values


sc_x = StandardScaler()
x_scaler = sc_x.fit_transform(x)
sc_y = StandardScaler()
y_scaler = sc_y.fit_transform(y.reshape(-1,1))




r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)
Z = x + 0.5
K = x - 0.4
plt.scatter(x, y, color="blue")
plt.plot(x,r_dt.predict(x),"red")


plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K), color="yellow")
print(r_dt.predict([[8.5]]))

# 0.6,0.8,0.9,1,1.5 da olsa aynı sonuca predict ediyor çünkü decision tree ağaca verileri koyuyor ve gelmiş olduğu aralığa göre aynı değeri döndürüyor.
