import pandas as pd
import matplotlib.pyplot as plt
import math
import random
datas = pd.read_csv("Ads_CTR_Optimisation.csv")

# her bir ilanın tıklanıp tıklanmadığına bakıcaz şayet tıklandı ise bu tıklama degerini dondurucez
N = 10000 # 10.000 tıklama
d = 10 # Toplam 10 ilan var
total = 0 # total prize
# Ri(n)
prizes = [0] * d # prizes 10 elemanlı bir liste olucak ve listenin her bir elemanı 0 olucak
# Ni(n)
chosenOnes = []  # secilenler
ones = [0] * d
zeros = [0] * d
for n in range(0,N):
    ad = 0 # secilen ilan
    max_th = 0
    for i in range(0,d):
        rasbeta = random.betavariate(ones[i]+1, zeros[i]+1)
        if(rasbeta > max_th):
            max_th = rasbeta
            ad = i
    chosenOnes.append(ad) # en yüksek ucb'ye sahip ilanı ekledik
    prize = datas.values[n,ad]
    if prize == 1:
        ones[ad] += 1
    else:
        zeros[ad] += 1
    prizes[ad] = prizes[ad] + prize 
    total += prize

print("Total prize: ",total)
plt.hist(chosenOnes)
