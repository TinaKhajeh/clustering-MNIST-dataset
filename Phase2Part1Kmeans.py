from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import h5py
import numpy
import sys

file = h5py.File('/Users/Tina/Documents/mnits0t5.h5', 'r')
x = file['x']
y = file ['y']

x = x.value
y = y.value

#Kmeans code which is commented to get resualts from other parts

zeroIndexes = x<128
OneIndexes = x>=128

#make binary data set
x[zeroIndexes] = 0
x[OneIndexes] = 1


xNew=[]
for index in range(0,6000):
    xNew.append(x[index].ravel());
    

k_meansClustring = KMeans(n_clusters=6);
k_meansClustring.fit(xNew)
yPredict = k_meansClustring.fit_predict(xNew)

microMeasure = f1_score(y[0:6000], yPredict, average='micro')  
macroMeasure = f1_score(y[0:6000], yPredict, average='macro') 
adjustedRIMeaasure = adjusted_rand_score(y[0:6000], yPredict) 

'''
#----------------------------------------------dim reduction using PCA: 
xNew2 =[]#for dim reduction
for index in range(0,6000):
    xNew2.append(x[index].ravel());




pca = PCA(n_components = 200)
xres = pca.fit_transform(xNew2)

#make binary data set


k_meansClustring2 = KMeans(n_clusters=6);
k_meansClustring2.fit(xres)
yPredict2 = k_meansClustring2.fit_predict(xres)
#shomare cluster ha ro dorost mikonim
for i in range (6000):
    if(yPredict2[i]==5):
        yPredict2[i]==1
    if(yPredict2[i]==1):
        yPredict2[i]==5
    if(yPredict2[i]==3):
        yPredict2[i]==4
    if(yPredict2[i]==4):
        yPredict2[i]==3   

microMeasure = f1_score(y[0:6000], yPredict2, average='micro')  
macroMeasure = f1_score(y[0:6000], yPredict2, average='macro') 
adjustedRIMeaasure = adjusted_rand_score(y[0:6000], yPredict2)


#----------------------------------------------dim reduction after binarization
'''