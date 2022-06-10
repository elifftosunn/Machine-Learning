import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

# data set
df = pd.read_csv("Churn_Modelling.csv",error_bad_lines=False)
X = df.iloc[:,3:13].values
y = df.iloc[:,13].values

# encoding of categorical data
labelEncode = LabelEncoder()
X[:,1] = labelEncode.fit_transform(X[:,1])
X[:,2] = labelEncode.fit_transform(X[:,2])

# 0-1-2-3 olan dataları OneHotEndoding ile encode etme
col_transformer = ColumnTransformer(
                        remainder="passthrough",
                        transformers = [
                            ("ohe",OneHotEncoder(dtype=float),slice(1,3))
                        ],
                )
X = col_transformer.fit_transform(X)

def heatmapPloth(cm,title,acc_score):
    plt.figure(figsize=(12,7))
    sns.heatmap(cm,annot=True,fmt=".2f")
    plt.title("{0} Accuracy Score:{1}".format(title,acc_score))
    plt.show()
    
# train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import pickle # diske kaydedilmiş olan modeli dosyaya yüklüyor ve o yüklemiş oldugu modeli aldı calıstirdi ve calismis halini test etti.
file = "model.register"
'''
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print("First Result",y_pred)
heatmapPloth(confusion_matrix(y_test,y_pred),"XGBClassifier", accuracy_score(y_test, y_pred))

import pickle
file = "model.register"
pickle.dump(classifier, open(file,"wb")) # yazma modunda binary olarak dosya aciliyor
'''
uploaded = pickle.load(open(file,"rb")) # read,binary
print("Second Result",uploaded.predict(X_test))
#import joblib => Bunlar da farklı model yukleme araclari
#import pmml

'''
# PCA'de amaç verileri birbirinden ayrıştıran en iyi algoritmayı bulmak oluyor.
from sklearn.decomposition import PCA
# import PCA
pca = PCA(n_components=2) # 2 column'a indirgeme
X_train_Pca = pca.fit_transform(X_train)
X_test_Pca = pca.transform(X_test)
    
    
# application of machine learning algorithm without pca
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
heatmapPloth(confusion_matrix(y_test,y_pred_log),"Logistic Reg.",accuracy_score(y_test,y_pred_log))

# application of machine learning algorithm with pca
log_reg_pca = LogisticRegression(random_state=0)
log_reg_pca.fit(X_train_Pca, y_train)
y_pred_pca = log_reg_pca.predict(X_test_Pca)
heatmapPloth(confusion_matrix(y_test,y_pred_pca),"LOGISTIC-PCA", accuracy_score(y_test, y_pred_pca))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import LDA
lda = LDA(solver="svd",n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.predict(X_test)

# application of machine learning algorithm with lda
log_reg_lda = LogisticRegression(random_state=0)
log_reg_lda.fit(X_train_lda, y_train)
y_pred_lda = log_reg_lda.predict(X_test_lda.reshape(-1,1))
heatmapPloth(confusion_matrix(y_test,y_pred_lda),"LOGISTIC-LDA", accuracy_score(y_test, y_pred_lda))

from sklearn.svm import SVC
# application of machine learning algorithm without pca
svm = SVC(kernel="rbf",random_state=0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
heatmapPloth(confusion_matrix(y_test,y_pred_svm),"SVC", accuracy_score(y_test, y_pred_svm))

# application of machine learning algorithm with pca
svm_pca = SVC(kernel="rbf",random_state=0)
svm_pca.fit(X_train_Pca, y_train)
y_pred_svm_pca = svm_pca.predict(X_test_Pca)
heatmapPloth(confusion_matrix(y_test,y_pred_svm_pca),"SVC-PCA", accuracy_score(y_test, y_pred_svm_pca))

# application of machine learning algorithm with lda
svm_lda = SVC(kernel="rbf",random_state=0)
svm_lda.fit(X_train_lda, y_train)
y_pred_svm_lda = svm_lda.predict(X_test_lda.reshape(-1,1))
heatmapPloth(confusion_matrix(y_test,y_pred_svm_lda),"SVC-LDA", accuracy_score(y_test, y_pred_svm_lda))

'''









