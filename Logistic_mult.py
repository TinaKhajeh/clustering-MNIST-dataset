from sklearn import svm
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import h5py
from sklearn import linear_model
file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')


xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 

'''
penaltiList = ['l1','l2']
CArray = [0.001,0.01,0.1,1,10,100] 

AccArr=[]
F1MacroArr=[]
F1MicroArr=[]
for pen in penaltiList:
    for c in CArray:
        LR = LogisticRegression(penalty=pen, C=c).fit(xtrain,ytrain)
        yPredict = LR.predict(xval)
        AccArr.append(accuracy_score(yval, yPredict))
        F1MacroArr.append(f1_score(yval, yPredict, average='macro'))
        F1MicroArr.append(f1_score(yval, yPredict, average='micro'))

'''
        
        
  
#best output = L2, c=10
LR = LogisticRegression(penalty='l2', C=10).fit(xtrain,ytrain)
y_test_Predict = LR.predict(xtest)
import pickle
file = open('multiClass_test_logistic', 'w')
pickle.dump(y_test_Predict, file)
file.close()

#to read the content of the file use command : 
f = open('multiClass_test_logistic', 'r')
yTestPredic = pickle.load(f)        
