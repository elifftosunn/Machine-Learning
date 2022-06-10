import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv("sepet.csv",header=None)
print(datas.isnull().sum())
from apyori import apriori
data = []
for i in range(0,7501):
    data.append([str(datas.values[i,j]) for j in range(0,20)])
rules = apriori(data,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
for i in rules:
    print(i,"\n")
    
#from eclat import eclat
#rules2 = eclat(data,min_support=0.01)
    
    
    