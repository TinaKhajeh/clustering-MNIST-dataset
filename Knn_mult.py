from sklearn import svm
import numpy
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import h5py
file = h5py.File('/Users/Tina/Documents/ml_proj1_data.h5', 'r')
file.keys()


xtest = file['xtest']
xtrain = file ['xtrain'] 
xval = file ['xval'] 
ytrain = file['ytrain']
yval = file ['yval'] 

#______________________________________feature selection:
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.03, penalty="l1", dual=False).fit(xtrain, ytrain)
model = SelectFromModel(lsvc, prefit=True)
X_train_new = model.transform(xtrain)
X_val_new = model.transform(xval)
X_test_new = model.transform(xtest)

'''
K1 = [5,10,20,25,50,100,250,500,1000,2500,5000,10000];
acc = []
f1MicroScore = []
f1MacroScore = []
for i in range(0,len(K1)):
    neigh = KNeighborsClassifier(n_neighbors=K1[i], weights= 'distance')
    neigh.fit(X_train_new, ytrain)
    yValPredict = neigh.predict(X_val_new)
    acc.append(accuracy_score(yval, yValPredict))
    f1MicroScore.append(f1_score(yval, yValPredict, average='micro'))
    f1MacroScore.append(f1_score(yval, yValPredict, average='macro'))
   

'''

neigh = KNeighborsClassifier(n_neighbors=20, weights= 'distance')
neigh.fit(X_train_new, ytrain)
y_testPredictKnn = neigh.predict(X_test_new)
 

import pickle
file = open('multiClass_test_knn', 'w')
pickle.dump(y_testPredictKnn, file)
file.close()

#to read the content of the file use command : 
f = open('multiClass_test_knn', 'r')
yTestPredic = pickle.load(f)
