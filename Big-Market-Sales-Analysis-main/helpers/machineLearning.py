import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve,auc 
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score,f1_score 

# SVM, RandomForestRegressor,GradientBoostingRegressor

class MachineLearning:
    def __init__(self,df):
        self.df = df
    def get_dataset(self,target,test_size,random_state):
        X = self.df.drop(target,axis=1)
        y = self.df[target]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        # if len(np.sum(y_train)) in [len(y_train),0]:
        #     print("all one class")
        return X_train,X_test,y_train, y_test   
    def standardScaler(self,train, test):
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)    
        return train,test
    def modelsCreateClassifier(self):
        classifiers = [
            ('LogisticReg', LogisticRegression(max_iter=10000)),
            ('KNN', KNeighborsClassifier()),
            ("SupportVectorMachines", SVC()),
            ("DecisionTree", DecisionTreeClassifier()),
            ("RandomForest", RandomForestClassifier()),
            ('Adaboost', AdaBoostClassifier()),
            ('GradientBoost', GradientBoostingClassifier()),
            ('XGBoost', XGBClassifier(
                use_label_encoder=False, eval_metric='logloss')),
            ('DecisionTree',DecisionTreeClassifier()),
            ('LightGBM', LGBMClassifier()),
        ('CatBoost', CatBoostClassifier(verbose=False))
        ]
        return classifiers
    def score(self,target, test_size, randomState):
        X_train,X_test,y_train,y_test = self.get_dataset(target, test_size, randomState)
        models = self.modelsCreateClassifier()
        datas = []
        names = []
        for name,model in models:
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            cv_results = cross_val_score(model,X_train, y_train, cv=10, scoring="accuracy")
            # cv=10 => train datasetini 10 parcaya boluyoruz 9 parcasında modeli kuruyor, 1 parcasında da modeli test ediyor
            #cv_mean = "%s: (%f)" % (name,cv_results.mean()) # Bu 10 sonucun ortalaması
            #acc_score = "%s: (%f)" % (name,accuracy_score(y_test,y_pred)) #tahminlerin dogruluk oranı
            cv_mean = cv_results.mean()
            acc_score = accuracy_score(y_test, y_pred)
            sns.heatmap(confusion_matrix(y_test, y_pred),annot=True,fmt=".2f",cmap="Greens")
            plt.title("{}: {}".format(name,acc_score))
            plt.show()
            datas.append([cv_mean,acc_score])
            names.append(name)
        resultDf = pd.DataFrame(index=names,data=datas,columns=["cv_mean","acc_score"])         
        return resultDf
    def modelCreateRegressor(self,target, test_size, random_state, cv, scoring):
        regressors = [('LinearReg', LinearRegression()),
                      ('PolynomialReg',PolynomialFeatures()),
                      ('KNN', KNeighborsRegressor()),
                      ("SupportVectorReg", SVR()),
                      ("DecisionTree", DecisionTreeRegressor()),
                      #("RandomForest", RandomForestRegressor()),
                      ('Adaboost', AdaBoostRegressor()),
                      ('GBM', GradientBoostingRegressor()),
                      ('XGBoost', XGBRegressor(
                          use_label_encoder=False)),
                      ('LightGBM', LGBMRegressor())#[LightGBM] [Fatal] Do not support special JSON characters in feature name.
                      ]
        X_train, X_test, y_train, y_test = self.get_dataset(target, test_size, random_state)
        for name,model in regressors:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_results = cross_val_score(model, X_train, y_train, cv = cv, scoring=scoring)
            cv_mean = cv_results.mean()
            acc_score = accuracy_score(y_test, y_pred)
            #self.confusion_matrix(y_test, y_pred)
            sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt=".2f",cmap="PuBuGn")
            plt.title(model)
            plt.show()
            return model, cv_mean, acc_score
    def modelResults(self,regressors_or_classifiers,target,test_size,random_state,cv,scoring):
        X_train, X_test, y_train, y_test = self.get_dataset(target, test_size, random_state)
        classifiers = self.modelCreateClassifier()
        regressors = self.modelCreateRegressor()
    def get_model(self,model,target,test_size,random_state,scoring, cv): # for the most suitable model
        X_train, X_test, y_train, y_test = self.get_dataset(target, test_size, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #self.confusion_matrix(y_pred,y_test)
        acc_score = "%s: (%f)" % (model,accuracy_score(y_test,y_pred)) #tahminlerin dogruluk oranı
        sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt=".2f",cmap="PuBuGn")
        plt.title(acc_score)
        plt.show()
        test_summary = classification_report(y_test, y_pred)
        X_pred = model.predict(X_train)
        train_summary = classification_report(y_train, X_pred)
        pre_score = precision_score(y_test, y_pred)
        fScore = f1_score(y_test, y_pred)
        return pre_score,fScore, test_summary, train_summary
    
    def hyperparameter_optimization(model, param_grid, target, test_size, random_state, scoring, cv): # for the most suitable model 
        X_train, X_test, y_train, y_test = self.get_dataset(target, test_size, random_state)
        gridSearchCV = GridSearchCV(model, param_grid ,scoring=scoring, n_jobs=-1,cv = cv)
        result = gridSearchCV.fit(X_train, y_train)
        bestParams = gridSearchCV.best_params_
        bestScore = gridSearchCV.best_score_
        return bestParams,bestScore
    
    def confusion_matrix(test,train):
        sns.heatmap(confusion_matrix(test,train), annot=True, fmt=".2f",cmap="Greens")
        plt.show()
    def generate_auc_roc_curve(model,X_test,y_test):
        probs = model.predict_proba(X_test)
        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()   
        
'''
    def modelCreateClassifier(self,target, test_size, random_state, cv, scoring):
        classifiers = [
            ('LogisticReg', LogisticRegression(max_iter=10000)),
            ('KNN', KNeighborsClassifier()),
            ("SupportVectorMachines", SVC()),
            ("DecisionTree", DecisionTreeClassifier()),
            ("RandomForest", RandomForestClassifier()),
            ('Adaboost', AdaBoostClassifier()),
            ('GBM', GradientBoostingClassifier()),
            ('XGBoost', XGBClassifier(
                use_label_encoder=False, eval_metric='logloss')),
            ('DecisionTree',DecisionTreeClassifier()),
            ('LightGBM', LGBMClassifier()),
        # ('CatBoost', CatBoostClassifier(verbose=False))
        ]
        X_train, X_test, y_train, y_test = self.get_dataset(target, test_size, random_state)
        for name,model in classifiers:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_results = cross_val_score(model, X_train, y_train, cv = cv, scoring=scoring)
            cv_mean = cv_results.mean()
            acc_score = accuracy_score(y_test, y_pred)
            #self.confusion_matrix(y_test, y_pred)
            sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt=".2f",cmap="PuBuGn")
            plt.title(model)
            plt.show()
            return model, cv_mean, acc_score

'''        
        
        
        
        
        
        
