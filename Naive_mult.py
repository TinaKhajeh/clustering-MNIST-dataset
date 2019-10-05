from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import h5py
file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')
file.keys()

xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 


gnb = GaussianNB()
y_pred = gnb.fit(xtrain, ytrain).predict(xval)
tmpAcc = accuracy_score(yval, y_pred)
tmpF1Macro = f1_score(yval, y_pred,average='macro')
tmpF1Micro = f1_score(yval, y_pred,average='micro')


y_pred_test = gnb.fit(xtrain, ytrain).predict(xtest)

#write to file
import pickle
file = open('multiClass_test_naiv', 'w')
pickle.dump(y_pred_test, file)
file.close()

#to read the content of the file use command : 
f = open('multiClass_test_naiv', 'r')
yTestPredic = pickle.load(f)