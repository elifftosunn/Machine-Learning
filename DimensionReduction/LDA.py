import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
# LDA'de amaç sınıfları birbirinden ayıran en iyi boyutu bulmak iken 
# LDA => Sınıfları göze alır, gözetimli öğrenmedir.

def heatmapPloth(cm,title,acc_score):
    plt.figure(figsize=(15,12))
    sns.heatmap(cm, annot = True, fmt = ".2f")
    plt.title("{0}:{1}".format(title,acc_score))
    plt.show()
    
df = pd.read_csv("wine.csv")

X = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lda = LDA(n_components=2,solver="svd") # solver : {'svd', 'lsqr', 'eigen'}
X_train_lda = lda.fit_transform(X_train,y_train) # LDA'nin calisabilmesi icin sınıfları ogrenmesi gerekiyor yani sınıflar arasındaki farkı maximize ediyor. PCA icin hangi sınıfta oldugu onemli degil.
X_test_lda = lda.transform(X_test) # burada tek parametre verebiliriz cunku ogrenilecek birsey yok egitimde yeni boyut olusturdu, o olusturulan yeni boyutlara gore dataları okuyor.

log_reg_lda = LogisticRegression(random_state=0)
log_reg_lda.fit(X_train_lda, y_train)
y_pred_lda = log_reg_lda.predict(X_test_lda)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

heatmapPloth(confusion_matrix(y_pred,y_pred_lda),"LDA Acc Score:", accuracy_score(y_pred, y_pred_lda))
heatmapPloth(confusion_matrix(y_test,y_pred),"LDA Acc Score:", accuracy_score(y_test, y_pred))
heatmapPloth(confusion_matrix(y_test,y_pred_lda),"LDA Acc Score:", accuracy_score(y_test, y_pred_lda))




