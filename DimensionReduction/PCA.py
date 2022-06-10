import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# DATA'YI SIKISTIRMAK İCİN DAHA AZ YER KAPLASIN DİYE PCO ALGORTHMASINI 
# KULLANDIK VE BU VERİDE 1/36 ORANINDA HATA YAPTI  

df = pd.read_csv("wine.csv")
#print(df.isnull().sum())

X = df.iloc[:,0:13].values
y = df.iloc[:,13].values

def heatmapPloth(cm,title,acc_score):
    sns.heatmap(cm, annot = True, fmt = ".2f")
    plt.title("{0}:{1}".format(title,acc_score))
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=(0))
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA  => Boyut İndirgeme
# PCA'de amaç verileri birbirinden ayrıştıran en iyi algoritmayı bulmak oluyor.
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,r2_score,accuracy_score,classification_report
pca = PCA(n_components=2) # n_components: kaç boyuta(2) indirgemesi gerektigini söyledik.
X_train2 = pca.fit_transform(X_train)
# fit eğitmek için transform ise o eğitimi veri kümesinde uygulamak için kullanılıyor.
X_test2 = pca.transform(X_test) # fit'den gelen egitilmis veriyi transform et
log_reg = LogisticRegression(random_state=0) # her log_regression'da burada verdigim sbt deger ile log_reg'i kullan.
log_reg.fit(X_train2, y_train)
y_pred = log_reg.predict(X_test2)
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
heatmapPloth(cm, "Actual and PDO", acc_score)
print("Actual Data and PCA Result",classification_report(y_test, y_pred))


log_reg2 = LogisticRegression(random_state=(0))
log_reg2.fit(X_train, y_train)
y_pred2 = log_reg2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
acc_score2 = accuracy_score(y_test, y_pred2)
heatmapPloth(cm2, "Actual and Not Pdo", acc_score2)

print("Actual Data and NOT PCA Result",classification_report(y_test, y_pred2))
print("Pca'li data and Pca'siz Data",classification_report(y_pred, y_pred2))
heatmapPloth(confusion_matrix(y_pred,y_pred2), "Pdo and Not Pdo", accuracy_score(y_pred, y_pred2))



from sklearn.svm import SVC
svm = SVC(kernel="sigmoid",C=0.1,gamma="scale")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_score_svm =  accuracy_score(y_test, y_pred_svm)
heatmapPloth(confusion_matrix(y_test,y_pred_svm), "SVC Accuracy",acc_score_svm)

# GridSearchCV: Model için en uygun parametreleri bize söylüyor.Yani modeli optimize ediyor.
from sklearn.model_selection import GridSearchCV
params = [{'kernel':["rbf","sigmoid","auto"],
           'C':[1,2,3,4,5],
           'gamma':range(1,100),
           'decision_function_shape':["ovo","ovr"]},
          {'kernel':["sigmoid","linear","rbf"],
           'C':[0.1,0.08,1,20,0.5],
           'gamma':["scale","auto"]}]
gsc = GridSearchCV(estimator = svm,
                   param_grid = params,
                   n_jobs=-1,
                   cv=10,
                   scoring="accuracy")
grid_search = gsc.fit(X_train, y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_
print(bestResult,"\n",bestParams)





