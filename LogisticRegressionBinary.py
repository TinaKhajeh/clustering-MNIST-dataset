from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import h5py
from sklearn import linear_model
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

penaltiList = ['l1','l2']
CArray = [0.0001,0.001,0.01,0.1,1,10,100,1000] 

'''
AccArr=[]
FArr=[]
for pen in penaltiList:
    for c in CArray:
        LR = LogisticRegression(penalty=pen, C=c).fit(xtrain,bin_train)
        yPredict = LR.predict(xval)
        AccArr.append(accuracy_score(bin_val, yPredict))
        FArr.append(f1_score(bin_val, yPredict))
'''

#best output = L2, c=10
LR = LogisticRegression(penalty='l2', C=10).fit(xtrain,bin_train)
y_test_Predict = LR.predict(xtest)
import pickle
file = open('bin_test_logistic', 'w')
pickle.dump(y_test_Predict, file)
file.close()

#to read the content of the file use command : 
f = open('bin_test_logistic', 'r')
yTestPredic = pickle.load(f)