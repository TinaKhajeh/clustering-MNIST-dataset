from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from predictEM import predictEM
from decimal import *
import h5py
import numpy
import sys

file = h5py.File('/Users/Tina/Documents/mnits0t5.h5', 'r')
x = file['x']
y = file ['y']

x = x.value
y = y.value

zeroIndexes = x<128
OneIndexes = x>=128
    
#make binary data set
x[zeroIndexes] = 0
x[OneIndexes] = 1

xNew2 =[]#for dim reduction
for index in range(0,6000):
    xNew2.append(x[index].ravel());



pca = PCA(n_components = 200)

pca.fit(xNew2)#bad nemudar bekesham az varance ha o tozihat 
xNew = pca.fit_transform(xNew2)
#make binary:


#Bernouli MM.
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
                tmp = Decimal(pow(mukd,xnd)*pow(1-mukd,1-xnd))*Decimal(tmp)#khob vaghean vase bazi halata in sefr mishe. !!! yek seri ha inghad kuchik mishan sefr mishan
        
            datas[nCount,kCount] = Decimal(pi[kCount])*Decimal(tmp);#surat             
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
microMeasure = f1_score(y[0:6000], predictY[0], average='micro')  
macroMeasure = f1_score(y[0:6000], predictY[0], average='macro') 
adjustedRIMeaasure = adjusted_rand_score(y[0:6000], predictY[0])


