import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,r2_score,accuracy_score,classification_report

def heatmapPloth(cm,title,acc_score):
    plt.figure(figsize=(15,12))
    sns.heatmap(cm, annot = True, fmt = ".2f")
    plt.title("{0}:{1}".format(title,acc_score))
    plt.show()
    
df = pd.read_csv("student-mat.csv")
#print(df.isnull().sum())
#print(df.info())


# kategorik dataların binaryzasyonu yapıldı
labelEncode = LabelEncoder()
df["school"] = labelEncode.fit_transform(df["school"])
df["sex"] = labelEncode.fit_transform(df["sex"])
df["address"] = labelEncode.fit_transform(df["address"])
df["famsize"] = labelEncode.fit_transform(df["famsize"])
df["Pstatus"] = labelEncode.fit_transform(df["Pstatus"])
df["Mjob"] = labelEncode.fit_transform(df["Mjob"])
df["Fjob"] = labelEncode.fit_transform(df["Fjob"])
df["reason"] = labelEncode.fit_transform(df["reason"])
df["guardian"] = labelEncode.fit_transform(df["guardian"])
df["schoolsup"] = labelEncode.fit_transform(df["schoolsup"])
df["famsup"] = labelEncode.fit_transform(df["famsup"])
df["paid"] = labelEncode.fit_transform(df["paid"])
df["activities"] = labelEncode.fit_transform(df["activities"])
df["nursery"] = labelEncode.fit_transform(df["nursery"])
df["higher"] = labelEncode.fit_transform(df["higher"])
df["internet"] = labelEncode.fit_transform(df["internet"])
df["romantic"] = labelEncode.fit_transform(df["romantic"])

X = df.iloc[:,:30].values
y = df.iloc[:,30].values # G1 values

# division of training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
# Feature Scaling (datas between 0-1) => Ölcekleme
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.decomposition import PCA
# PCA => Boyut İndirgeme
'''
    PCA'de amaç verileri birbirinden ayrıştıran en iyi algoritmayı bulmak oluyor.
	PCA => Sınıf farkı yoktur, gözetimsiz bir algoritmadır.
'''
pca = PCA(n_components=4)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
heatmapPloth(confusion_matrix(y_test,y_pred),"Actual and No PDO", accuracy_score(y_test, y_pred))

log_reg2 = LogisticRegression(random_state=0)
log_reg2.fit(X_train2, y_train) 
y_pred2 = log_reg2.predict(X_test2)
heatmapPloth(confusion_matrix(y_test,y_pred2),"Actual and PDO" , accuracy_score(y_test, y_pred2))

heatmapPloth(confusion_matrix(y_pred,y_pred2),"Pca'li and Pca'siz Data", accuracy_score(y_pred,y_pred2))

'''
    LDA'de amaç sınıfları birbirinden ayıran en iyi boyutu bulmak 
	LDA => Sınıfları göze alır, gözetimli öğrenmedir.
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver="svd",n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
log_reg_lda = LogisticRegression(random_state=0)
log_reg_lda.fit(X_train_lda,y_train)
y_pred_lda = log_reg_lda.predict(X_test_lda)
heatmapPloth(confusion_matrix(y_test,y_pred_lda),"Actual and LDA", accuracy_score(y_test,y_pred_lda))

'''
Cross-validation, makine öğrenmesi modelinin görmediği veriler üzerindeki performansını
mümkün olduğunca objektif ve doğru bir şekilde değerlendirmek için kullanılan 
istatistiksel bir yeniden örnekleme(resampling) yöntemidir.
'''
from sklearn.model_selection import cross_val_score
success = cross_val_score(estimator = log_reg , X = X_train, y = y_train, cv = 4)
print(success.mean()) # 0.10606060606060606
print(success.std())  # 0.015151515151515152

# Model icin herhangi bir algoritma kullanıyoruz.
from sklearn.svm import SVC
svc = SVC(kernel="rbf",C=3,gamma=0.06)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
heatmapPloth(confusion_matrix(y_test,y_pred_svc),"SVC Accuracy",accuracy_score(y_test, y_pred_svc))


'''
# parameter optimization and algorithm selection
from sklearn.model_selection import GridSearchCV
params = [{'kernel':["linear","rbf","sigmoid"],
           'C':[1,2,3,4,5],
           'gamma':["scale","auto"],
           'decision_function_shape':["ovo","ovr"]},
          {'kernel':["rbf","poly","linear"],
           'gamma':["scale","auto"],
           'C':[1,10,100,1000]},
          {'kernel':["rbf","sigmoid"],
           'C':range(1,100),
           'gamma':np.arange(1e-4,1e-2,0.0001)}]
gs = GridSearchCV(estimator = svc, param_grid=params,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
grid_search = gs.fit(X_train,y_train)
bestResult = grid_search.best_score_
bestParams = grid_search.best_params_
print(bestResult,"\n",bestParams)
'''


plt.figure(figsize=(20,18))
sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.show()





