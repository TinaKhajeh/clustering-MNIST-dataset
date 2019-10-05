from sklearn import svm
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import h5py
file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')
file.keys()
xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 

#______________________________________Feature selection 
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.05, penalty="l1", dual=False).fit(xtrain, ytrain)
model = SelectFromModel(lsvc, prefit=True)
X_train_new = model.transform(xtrain)
X_val_new = model.transform(xval)
X_test_new = model.transform(xtest)


#_______________________________________scale all inputs : train,val & test in range [-1,1] for svm
from sklearn import preprocessing
import numpy as np
max_abs_scaler = preprocessing.MaxAbsScaler()
xtrainScaled = max_abs_scaler.fit_transform(X_train_new)

max_abs_scaler2 = preprocessing.MaxAbsScaler()
xvalScaled= max_abs_scaler2.fit_transform(X_val_new)

max_abs_scaler3 = preprocessing.MaxAbsScaler()
xtestScaled = max_abs_scaler3.fit_transform(X_test_new)


#linear svm : 
''''
Carray = [0.0001,0.001,0.01,0.1,1,10]
AccArray =[]
F1MacroArray=[]
F1MicroArray=[]
for c in Carray:
    lin_clf = svm.LinearSVC(C=c)
    lin_clf.fit(X_train_new, ytrain)
    yvalPredict = lin_clf.predict(X_val_new)
    AccArray.append(accuracy_score(yval, yvalPredict))
    F1MacroArray.append(f1_score(yval, yvalPredict, average='macro'))
    F1MicroArray.append(f1_score(yval, yvalPredict, average='micro'))
'''
'''
#RBF svm
Carray = [0.0001,0.001,0.01,0.1,1,10,100,1000]
AccArray =[]
F1MacroArray=[]
F1MicroArray=[]
for c in Carray:
    RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = 0.1)
    RBFSVM.fit(xtrainScaled,ytrain)
    yValPredict2 = RBFSVM.predict(xvalScaled)
    AccArray.append(accuracy_score(yval, yValPredict2))
    F1MacroArray.append(f1_score(yval, yValPredict2, average='macro'))
    F1MicroArray.append(f1_score(yval, yValPredict2, average='micro'))
'''    
'''    
c= 10
gammaArray = [0.00001,0.0001,0.001,0.01,0.1,1,10]
AccArray =[]
F1MacroArray=[]
F1MicroArray=[]
for g in gammaArray:
    RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = g)
    RBFSVM.fit(xtrainScaled,ytrain)
    yValPredict2 = RBFSVM.predict(xvalScaled)
    AccArray.append(accuracy_score(yval, yValPredict2))
    F1MacroArray.append(f1_score(yval, yValPredict2, average='macro'))
    F1MicroArray.append(f1_score(yval, yValPredict2, average='micro'))    
'''
'''
c=1
g= 0.1
RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = g)
RBFSVM.fit(xtrainScaled,ytrain)
yValPredict2 = RBFSVM.predict(xvalScaled)
acc=accuracy_score(yval, yValPredict2)
f1Macro = f1_score(yval, yValPredict2, average='macro')
f1Micro = f1_score(yval, yValPredict2, average='micro')


'''

c=10
g= 1
RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = g)
RBFSVM.fit(xtrainScaled,ytrain)
yValPredictTest = RBFSVM.predict(xtestScaled)

import pickle
file = open('multiClass_test_svm', 'w')
pickle.dump(yValPredictTest, file)
file.close()

#to read the content of the file use command : 
f = open('multiClass_test_svm', 'r')
yTestPredic = pickle.load(f)

