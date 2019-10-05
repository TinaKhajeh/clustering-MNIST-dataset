from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import math
import h5py

file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')
file.keys()

xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 
reg_val = file['reg_val']

with open('reg_train.txt') as f:
    content = f.read();
    
content = content.split('\n')
content = content[0:17000]
reg_train = map(float,content)
'''
kArray = [10,20,50,100]
RMSEArray = []
MAEArray = []
for k in kArray:
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(xtrain, reg_train)
    yPredic = neigh.predict(xval)
    RMSEArray.append(math.sqrt(mean_squared_error(reg_val, yPredic)))
    MAEArray.append(mean_absolute_error(reg_val, yPredic))    
'''    
    
    
    
neigh = KNeighborsRegressor(n_neighbors=20)
neigh.fit(xtrain, reg_train)
yPredic = neigh.predict(xtest)  

import pickle
file = open('reg_test_KNNReg', 'w')
pickle.dump(yPredic, file)
file.close()

#to read the content of the file use command : 
f = open('reg_test_KNNReg', 'r')
yTestPredic = pickle.load(f)