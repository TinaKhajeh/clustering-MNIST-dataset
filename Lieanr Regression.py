from sklearn import svm
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


#ridge
'''
alphaArray = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
RMSEArray = []
MAEArray = []
for alph in alphaArray:
    clf = linear_model.Ridge (alpha = alph)
    clf.fit (xtrain,reg_train) 
    yPredict = clf.predict(xval)
    RMSEArray.append(math.sqrt(mean_squared_error(reg_val, yPredict)))
    MAEArray.append(mean_absolute_error(reg_val, yPredict))
'''    
    
#lasso
'''
alphaArray = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
RMSEArray = []
MAEArray = []
for alph in alphaArray:
    clf = linear_model.Lasso (alpha = alph)
    clf.fit (xtrain,reg_train) 
    yPredict = clf.predict(xval)
    RMSEArray.append(math.sqrt(mean_squared_error(reg_val, yPredict)))
    MAEArray.append(mean_absolute_error(reg_val, yPredict))
'''
   
#polynomail
'''
poly = PolynomialFeatures(degree=2)
x_new = poly.fit_transform(xtrain)
clf = linear_model.Ridge (alpha = alph)
clf.fit (x_new,reg_train) 
yPredict = clf.predict(xval)
'''

clf = linear_model.Ridge (alpha = 1)
clf.fit (xtrain,reg_train)
yPredictTest = clf.predict(xtest)

import pickle
file = open('reg_test_LinearReg', 'w')
pickle.dump(yPredictTest, file)
file.close()

#to read the content of the file use command : 
f = open('reg_test_LinearReg', 'r')
yTestPredic = pickle.load(f)
