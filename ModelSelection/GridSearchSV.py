import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


datas = pd.read_csv("Social_Network_Ads.csv")
X = datas.iloc[:,2:4].values
y = datas.iloc[:,4].values

# division of training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=(0))
# Scaling(ölcekleme)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svm)
success = cross_val_score(estimator=svm, X=X_train,y=y_train,cv=4)
print(success.mean(),"\n",success.std())

# parameter optimization and algorithm selection
from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':["rbf","random","linear"]},
     {'C':[1,10,100,1000],'kernel':["rbf","linear"],
      'gamma':[1,0.5,0.1,0.01,0.06],
      'decision_function_shape':["ovo","ovr"]},
     {'C':[1,2,3,4,5],'kernel':["poly","sigmoid"],
      'gamma':[1,0.5,0.1,0.01,0.06]}]
'''
    GS parameter
    estimator: sınıflandırma algoritması(neyi optimize etmek istedigimiz)
    param_grid: parametreler/ denenecekler
    scoring: neye göre skorlanacak for example: accuracy
    cv: kaç katlamalı olacağı (kaç katlamada veriyi kontrol edeceği)
    n_jobs: aynı anda çalışacak iş
'''

gs = GridSearchCV(estimator = svm ,
                  param_grid = p,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
grid_search = gs.fit(X_train, y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_
print(bestResult,"\n",bestParams)

from sklearn.model_selection import cross_val_score
success = cross_val_score(estimator = svm , X = X_train, y = y_train, cv = 4)
print(success.mean(),"\n",success.std())
# mean: 0.9067164179104478                  ///Overfitting olayı yok ve model basarili
# standard sapma:  0.030542361089076302
'''
Cross Validation bizi hem overfitting problemiyle karşı karşıya olup olmadığımızı hem de modelimizin
 kalitesini görmemizi sağlayacaktır. Böylece henüz görmediğimiz test veri setinde yüksek hata 
 oranları ile karşılaşmadan önce modelimizin performansı test etmemizi sağlayacaktır. 
 Uygulanabilirliği kolay olduğu için de sıklıkla kullanılan bir yöntemdir.
'''



