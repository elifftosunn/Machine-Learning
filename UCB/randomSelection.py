import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv("Ads_CTR_Optimisation.csv")

import random 
N = 10000
d = 10
total  = 0
chosenOnes = [] # secilmis olanlar
for n in range(0,N):
    ad = random.randrange(d) # 10'a(10 dahil) kadar olan rastgele tamsayı üret(columns sayısı)
    chosenOnes.append(ad)
    prize = datas.values[n,ad] # satır sayısı(n) ve kullanıcıdan gösterilen deger hangisi ise(ad), For example: [7,2] 7.indexteki satır 2.column degeri:1
    total = total + prize # her seferinde prize olarak tıklanacak deger bize prize olarak docenek
    
plt.hist(chosenOnes)
print(chosenOnes)
    

















