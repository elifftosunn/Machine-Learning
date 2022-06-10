import pandas as pd
import matplotlib.pyplot as plt

datas = pd.read_csv("Ads_CTR_Optimisation.csv")
# Random Selection (Rastgele Secim)
'''
import random
N = 10000 # row
d = 10 # columns
total = 0 
chosenOnes = []  # 0-1-1-1-0-0-1 gibi...
for n in range(0,N):
    ad = random.randrange(10) # 0-3-5-10-7-4.....
    chosenOnes.append(ad)
    prize = datas.values[n,ad]  # row,column
    total += prize
    
    
plt.hist(chosenOnes)
# tıklama ve o tıklama sonucunda dönecek ödül değeri

'''
import math
# her bir ilanın tıklanıp tıklanmadığına bakıcaz şayet tıklandı ise bu tıklama degerini dondurucez
N = 10000 # 10.000 tıklama
d = 10 # Toplam 10 ilan var
total = 0 # total prize
# Ri(n)
prizes = [0] * d # prizes 10 elemanlı bir liste olucak ve listenin her bir elemanı 0 olucak
# Ni(n)
clicks = [0] * d # o ana kadar ki tıklamalar
chosenOnes = []  # secilenler
for n in range(0,N):
    ad = 0 # secilen ilan
    max_ucb = 0
    for i in range(0,d):
        if (clicks[i] > 0):
            average = prizes[i] / clicks[i]
            delta = math.sqrt(3/2*math.log(n)/clicks[i])
            ucb = average + delta 
        else:
            ucb = N * 10
        if max_ucb < ucb: # greater than max the ad is out(max'tan daha büyük bir ilan çıktı.)
            max_ucb = ucb     
            ad = i
    chosenOnes.append(ad) # en yüksek ucb'ye sahip ilanı ekledik
    clicks[ad] += 1
    prize = datas.values[n,ad]
    prizes[ad] = prizes[ad] + prize 
    total += prize

print("Total prize: ",total)
plt.hist(chosenOnes)



