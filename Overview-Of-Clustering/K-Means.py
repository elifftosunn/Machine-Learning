import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv("customers.csv")

x = datas.iloc[:,3:].values

# kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(x)
print(kmeans.cluster_centers_)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=123)    
    kmeans.fit(x)
    # kmeans.inertia_: her bir çalıştırmadaki wss değeri
    wcss.append(kmeans.inertia_) # kmeans'in nekadar başarılı olduğu

    
plt.plot(range(1,11),wcss)
plt.show()
kmeans = KMeans(n_clusters=4,init="k-means++",random_state=123)
y_pred = kmeans.fit_predict(x)
plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1], s=100, c="r")
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1], s=100, c="y")
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1], s=100, c="b")
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1], s=100, c="g")
plt.title("KMeans Algorithm")
plt.show()

# HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, linkage="ward", affinity="euclidean")
y_pred2 = ac.fit_predict(x)

plt.scatter(x[y_pred2 == 0,0], x[y_pred2 == 0,1] , s=100 ,c="r")
plt.scatter(x[y_pred2 == 1,0], x[y_pred2 == 1,1] , s=100 ,c="y")
plt.scatter(x[y_pred2 == 2,0], x[y_pred2 == 2,1] , s=100 ,c="b")
plt.title("AgglomerativeClustering Algorithm")
plt.show()  

# Scipy library => görsellestirme icin
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()



