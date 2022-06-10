import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
comments = pd.read_csv("Restaurant-Reviews.csv",on_bad_lines='skip')
nullValues = comments.isnull().sum()
comments = comments.dropna(how="any")
comments.index = range(704);
nullValues2 = comments.isnull().sum()
# PREPROCESSING (Ön İşleme)
import re
import nltk
from nltk.stem.porter import PorterStemmer # kelimeyi koklerine ayırma
ps = PorterStemmer()
nltk.download('stopwords') # stop kelimeleri ayırma
from nltk.corpus import stopwords
compileComments = []
for i in range(704):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i]) # a'dan z'ye kadar olan harfler dışında .!,... gibi ifadeler içeren kelimeler yerine onları sil boşluk koy
    comment = comment.lower() # kucuk harfe cevrildi
    comment = comment.split() # bosluklar ayrıldı
    # all english stopwords'leri alıyoruz set ile kümeye ceviriyoruz şayet o kelime yoksa ozaman o kelimeyi stemle(yani gövdesini bul)
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    compileComments.append(comment)
    #print(comment)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

# Feature Extraction (Öznitelik Çıkarımı)
# Bag of Words (BOW)
cv = CountVectorizer(max_features=2000) # en fazla kullanılan 1000 kelimeyi al
X = cv.fit_transform(compileComments).toarray()   # bagımsız degisken (hem ogreniyor hem donusturuyor) # comments'ler sayısal degerlere donustu 
y = comments.iloc[:,1].values    # bagimli degisken


# Machine Learning
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def accuracyScoreGraph(cm,title,score):  
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('{0}-Accuracy Score: {1}'.format(title,score), size = 15)
    plt.show()


gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
score = gnb.score(x_test, y_test) # 1.way predict result
cm = confusion_matrix(y_test, y_pred) # 2.way predict result
cr = classification_report(y_test,y_pred) # binary values alıyor.######
print(classification_report(y_test, y_pred))
accuracyScoreGraph(cm,"GaussianNB",score)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred2 = lr.predict(x_test)    
score2 = lr.score(x_test, y_test)
cm2 = confusion_matrix(y_test, y_pred2)
print(classification_report(y_test, y_pred2))
accuracyScoreGraph(cm2,"LogisticRegression",score2)

knc = KNeighborsClassifier(n_neighbors=1,weights="distance",algorithm="ball_tree",metric="minkowski",p=2,leaf_size=30) #  weights : {'uniform', 'distance'}, algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, metric="precomputed"        
knc.fit(x_train, y_train)
y_pred3 = knc.predict(x_test)
cm3 = confusion_matrix(y_test,y_pred3)
score3 = knc.score(x_test, y_test)
print(classification_report(y_test, y_pred3))
accuracyScoreGraph(cm3,"KNN",score3)

dtc = DecisionTreeClassifier(random_state=42,criterion="entropy",splitter="random",max_features="auto") # criterion : {"gini", "entropy"}, splitter : {"best", "random"}, max_features : int, float or {"auto", "sqrt", "log2"}
dtc.fit(x_train, y_train)
y_pred4 = dtc.predict(x_test)
cm4 = confusion_matrix(y_test, y_pred4)
score4 = dtc.score(x_test, y_test)
print(classification_report(y_test, y_pred4))
accuracyScoreGraph(cm4,"DecisionTreeClassifier",score4)


svc = SVC(kernel="rbf",gamma="scale",cache_size=100,decision_function_shape="ovr") # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'},  gamma : {'scale', 'auto'}, decision_function_shape : {'ovo', 'ovr'}
svc.fit(x_train, y_train)
y_pred5 = svc.predict(x_test)
score5 = svc.score(x_test, y_test)
cm5 = confusion_matrix(y_test, y_pred5)
print(classification_report(y_test, y_pred5))
accuracyScoreGraph(cm5,"SVC",score5)

rfc = RandomForestClassifier(n_estimators=8,criterion="entropy",max_depth=20, max_features="auto",class_weight="balanced_subsample",random_state=0) # criterion : {"gini", "entropy"}, max_features : {"auto", "sqrt", "log2"}, class_weight : {"balanced", "balanced_subsample"}
rfc.fit(x_train, y_train)
y_pred6 = rfc.predict(x_test)
score6 = rfc.score(x_test, y_test)
cm6 = confusion_matrix(y_test, y_pred6)
print(classification_report(y_test, y_pred6))
accuracyScoreGraph(cm6,"RandomForestClassifier",score6)





'''
kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(x_train)
print(kmeans.cluster_centers_)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=123)    
    kmeans.fit(x_train)
    # kmeans.inertia_: her bir çalıştırmadaki wss değeri
    wcss.append(kmeans.inertia_) # kmeans'in nekadar başarılı olduğu


plt.plot(range(1,11),wcss)
plt.show()
kmeans = KMeans(n_clusters=4,init="k-means++",random_state=123)
y_pred6 = kmeans.fit_predict(x_train)
plt.scatter(x_train[y_pred6 == 0,0], x_train[y_pred6 == 0,1], s=100, c="r")
plt.scatter(x_train[y_pred6 == 1,0], x_train[y_pred6 == 1,1], s=100, c="y")
plt.scatter(x_train[y_pred6 == 2,0], x_train[y_pred6 == 2,1], s=100, c="b")
plt.scatter(x_train[y_pred6 == 3,0], x_train[y_pred6 == 3,1], s=100, c="g")
plt.title("KMeans Algorithm")
plt.show()

'''













