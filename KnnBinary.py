from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
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



    

#K1 = [10,100,1000,10000];
#acc = [0,0,0,0]
#fScore = [0,0,0,0]
#for i in range(0,len(K1)-1):
#    neigh = KNeighborsClassifier(n_neighbors=K1[i], weights= 'distance')
#    neigh.fit(xtrain, bin_train)
#    yValPredict = neigh.predict(xval)
#    acc[i] = accuracy_score(yval, yValPredict)
#    fScore[i] = f1_score(yval, yValPredict)



K1 = [5,10,25,50,100,250,500,1000,2500,5000,10000];
acc = []
fScore = []
for i in range(0,len(K1)):
    neigh = KNeighborsClassifier(n_neighbors=K1[i], weights= 'distance')
    neigh.fit(X_train_new, bin_train)
    yValPredict = neigh.predict(X_val_new)
    acc.append(accuracy_score(bin_val, yValPredict))
    fScore.append(f1_score(bin_val, yValPredict))
    
 
    
neigh = KNeighborsClassifier(n_neighbors=20, weights= 'distance')
neigh.fit(X_train_new, bin_train)
y_testPredictKnn = neigh.predict(X_test_new)
 

import pickle
file = open('bin_test_knn', 'w')
pickle.dump(y_testPredictKnn, file)
file.close()

#to read the content of the file use command : 
f = open('bin_test_knn', 'r')
yTestPredic = pickle.load(f)