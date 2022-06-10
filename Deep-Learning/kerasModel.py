import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,confusion_matrix,r2_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

datas = pd.read_csv("Churn_Modelling.csv",on_bad_lines='skip')
#print(datas.columns)
#print(datas.isnull().sum()) # no missing value


plt.figure(figsize=(15,15))
sns.heatmap(datas.corr(), annot=True, fmt=".2f",cbar=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

X = datas.iloc[:,3:-1].values
Y = datas.iloc[:,-1:].values

def heatmapGraph(cm,title,error):
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.title("{0}:{1}".format(title,error))
    plt.show()


# categoric datas's encodes
labelEncode = LabelEncoder()
X[:,1] = labelEncode.fit_transform(X[:,1])
ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")
X = ohe.fit_transform(X)
X[:,4] = labelEncode.fit_transform(X[:,4])


# separation of data into test and train
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=(0),shuffle=(True))
sc = StandardScaler() # Feature Scaling 
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
lr = LogisticRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
cm_log_reg = confusion_matrix(y_test, y_pred_lr)
mse_log_reg = mean_squared_error(y_test, y_pred_lr)
mae_log_reg = mean_absolute_error(y_test, y_pred_lr)
r2_score_lr = r2_score(y_test,y_pred_lr)
heatmapGraph(cm_log_reg, "LogisticRegression", mae_log_reg)


# regression algorithms for machine learning
knn = KNeighborsClassifier(n_neighbors=1,weights="uniform",
                           algorithm="auto",metric="minkowski").fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
heatmapGraph(cm_knn, "KNN", mae_knn)


svm = SVC(kernel="rbf",degree=4,C=1,gamma="scale",
          decision_function_shape="ovo").fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
heatmapGraph(cm_svm, "SVC", mae_svm)


kmeans = KMeans(n_clusters=1,init="k-means++",algorithm="elkan",n_init=20,max_iter=300).fit(X_train,y_train)
y_pred_kmeans = kmeans.predict(X_test)
cm_kmean = confusion_matrix(y_test, y_pred_kmeans)
mae_kmeans = mean_absolute_error(y_test, y_pred_kmeans)
heatmapGraph(cm_kmean, "KMeans", mae_kmeans)

'''
# create model
model = Sequential()
# add model layers
model.add(Dense(6,activation="relu",input_dim=12)) # input layer
model.add(Dropout(0.5))
model.add(Dense(3,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid")) # output layer
# compile model using mse as a measure of model performance
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['acc'])
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=20)
# train model
history = model.fit(x = X_train,y = y_train, validation_data=(X_test,y_test),
          batch_size=50,epochs=300,shuffle=True,callbacks=[earlyStopping])
#lostData = pd.DataFrame(model.history.history)


# Plot training & validation accuracy values
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],"ro")
plt.plot(history.history['val_loss'],"b*")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

# Error rate results
y_pred = (model.predict(X_test) > 0.5).astype("int32")
maeModel = mean_absolute_error(y_test,y_pred)
mseModel = mean_squared_error(y_test, y_pred)
cmModel = confusion_matrix(y_test,y_pred)
heatmapGraph(cmModel, "Model", maeModel)
print(classification_report(y_test, y_pred))
'''

'''
    
            Classification Report 
            
            Epoch 00118: early stopping
                          precision    recall  f1-score   support

                       0       0.81      1.00      0.89      5255  // 0 => %81 true result
                       1       0.91      0.08      0.14      1345  // 1 => %91 true result

                accuracy                           0.81      6600
               macro avg       0.86      0.54      0.52      6600
            weighted avg       0.83      0.81      0.74      6600
    
    
'''










