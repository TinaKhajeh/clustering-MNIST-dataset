
from sklearn.cluster import KMeans

from sklearn.metrics import f1_score
import h5py
import numpy
import sys
from predictEM import predictEM
from sklearn.metrics import adjusted_rand_score


file = h5py.File('/Users/Tina/Documents/mnits0t5.h5', 'r')
x = file['x']
y = file ['y']

x = x.value
y = y.value


myX = x[0:6000]
myY = y[0:6000]


zeroIndexes = myX<128
OneIndexes = myX>=128


#Binarization
myX[zeroIndexes] = 0
myX[OneIndexes] = 1

xNew=[]
for index in range(0,len(myX)):
    xNew.append(myX[index].ravel());


# Bernoli Mixture Modle : 
import numpy;
import random;
k=6;
#initialization:
nData = len(xNew)
d= len(xNew[1])#calculate feature size
pi = numpy.empty(k);
pi.fill(1.0/k);
mu = numpy.zeros((k, d))
for kCount in range(k):#initialize mu[k*features] array
    for dCount in range(d):
        mu[kCount,dCount] = random.uniform(0.25,  0.75);
    mu[kCount,] = mu[kCount,]/sum(mu[kCount,])
datas = numpy.zeros((nData, k))#for each data shows percentage of belong to k will compute and fill every iteration
Nk = numpy.zeros(k)#effective number of data points
for iteration in range(7):#number of execution
    print iteration
    print '=========================='
   #E_Step:
    for nCount in range(nData):#nData#for each data we want to fill datas which shows the percentage of belonging to kth mixture
        if (nCount%10 ==0):
            print nCount
        nThData = xNew[nCount]#select nth data
        for kCount in range(k):
            tmp = 1.0;
            muk = mu[kCount,];
            for dCount in range(d):# in zarb ro bordari anjam bede
                mukd = muk[dCount];
                xnd = nThData[dCount]
                tmp = (pow(mukd,xnd)*pow(1-mukd,1-xnd))*tmp#khob vaghean vase bazi halata in sefr mishe. !!! yek seri ha inghad kuchik mishan sefr mishan
        
            datas[nCount,kCount] = pi[kCount]*tmp;#surat             
        tmpSum = sum(datas[nCount,])   
        #sys.stdout.flush(); 
        if(tmpSum != 0):
            datas[nCount,] = (datas[nCount,]*1.0)/(tmpSum*1.0);
    print'****************';  
   #M_Step:
   #update Nk,mu,pi

    for kIndex in range(k):
        effNum = 0;
        avrageX =numpy.zeros(d);
        for index in range(nData):
            effNum = effNum+datas[index,kIndex]
            avrageX = datas[index,kIndex]*xNew[index]+avrageX
        Nk[kIndex] = effNum;
        mu[kIndex] = (1.0/effNum)*avrageX;
        pi[kIndex] = effNum/nData

#Prediction:
predictY = predictEM(datas)

#evaluation: 
microMeasure = f1_score(myY, predictY[0], average='micro')  
macroMeasure = f1_score(myY, predictY[0], average='macro') 
adjustedRIMeaasure = adjusted_rand_score(myY, predictY[0])