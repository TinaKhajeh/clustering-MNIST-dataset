from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import h5py
file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')
file.keys()
bin_train = file['bin_train']
bin_val = file['bin_val']
mult_train = file['mult_train']
mult_val = file['mult_val']
xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 

#______________________________________feature selection:
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.02, penalty="l1", dual=False).fit(xtrain, bin_train)
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




gammaArray = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1]
Carray = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]

#_____________________________________Linear SVM:
#linearAccuracy=[]
#linearFmeasure=[]
#for c in Carray:
#    LinearSVM = svm.SVC(C=c,kernel='linear')
#    linearSVM.fit(xtrainScaled,bin_train)
#    yValPredict = linearSVM.predict(xvalScaled)
#    tmpAcc = accuracy_score(yval, yValPredict)
#    tmpF = f1_score(yval, yValPredict)
#    linearAccuracy.append(tmpAcc)
#    linearFmeasure.append(tmpF)
    




#_______________rbf SVM:
#________________calculate C
rbfAccuracy=[]
rbfFmeasure=[]
for i in range(0,10):
        c=Carray[i]
        g=0.1
        RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = g)
        RBFSVM.fit(xtrainScaled,bin_train)
        yValPredict2 = RBFSVM.predict(xvalScaled)
        tmpAcc = accuracy_score(bin_val, yValPredict2)
        tmpF = f1_score(bin_val, yValPredict2)
        rbfAccuracy.append(tmpAcc)
        rbfFmeasure.append(tmpF)

c=Carray[7]
#gama:
rbfAccuracy=[]
rbfFmeasure=[]
for i in range(0,7):
        g=gammaArray[i]
        RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = g)
        RBFSVM.fit(xtrainScaled,bin_train)
        yValPredict2 = RBFSVM.predict(xvalScaled)
        tmpAcc = accuracy_score(bin_val, yValPredict2)
        tmpF = f1_score(bin_val, yValPredict2)
        rbfAccuracy.append(tmpAcc)
        rbfFmeasure.append(tmpF)
        
        
        
RBFSVM = svm.SVC(C=c,kernel='rbf',gamma = 1)
RBFSVM.fit(xtrainScaled,bin_train)
ytestPredict = RBFSVM.predict(xtestScaled)

import pickle
file = open('bin_test_svm', 'w')
pickle.dump(ytestPredict, file)
file.close()

#to read the content of the file use command : 
f = open('bin_test_svm', 'r')
yTestPredic = pickle.load(f)


