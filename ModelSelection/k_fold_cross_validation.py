import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# data sets
datas = pd.read_csv("Social_Network_Ads.csv")
#print(datas.isnull().sum())
print(datas.corr())
X = datas.iloc[:,2:4].values
y = datas.iloc[:,4].values

sns.countplot(datas["Purchased"])
plt.title("Purchased Number")
plt.show()
sns.scatterplot(datas["Age"], datas["EstimatedSalary"])
plt.show()
sns.distplot(datas["EstimatedSalary"])
plt.show()
def heatmapPloth(cm,title):
    sns.heatmap(cm, annot = True, fmt = ".2f")
    plt.title("{0}".format(title))
    plt.show()

# division of training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
# scaling (ölçekleme)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM 
svm = SVC(kernel="rbf",degree=3,gamma="scale",random_state=0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
heatmapPloth(cm_svm, "SVM")


# K-Katlamali çapraz doğrulama(K-Fold cross-validation)
from sklearn.model_selection import cross_val_score
'''
    1. estimator: svm gibi hangi algoritmayı kullanacagı
    2. X
    3. Y
    4. cv: kaç katlamalı
'''
successfull = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 4)
print(successfull.mean())
print(successfull.std()) # standard sapma nekadar düsük ise okadar iyidir.


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=(0))
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
heatmapPloth(confusion_matrix(y_test,y_pred_log), "Log.Regression")
successfull_log = cross_val_score(estimator=log_reg,X = X_train, y = y_train, cv = 4)
print(successfull_log.mean(),"\n",successfull_log.std())


 








