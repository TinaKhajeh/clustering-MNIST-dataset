from sklearn import svm
from scipy.sparse import csr_matrix
import h5py
import numpy
import sys
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score


#Simple PCA
file = h5py.File('/Users/Tina/Documents/csr_ml2.h5', 'r')
data = file['data']
indices = file ['indices']
indptr = file['indptr']
shape = file['shape']

labels = file['labels'].value

SData = csr_matrix((data, indices, indptr), shape=shape).toarray()


numData = len(SData)
kf = KFold(numData, n_folds=3)
accScore = 0;
RBFSVM = svm.SVC(C=1,kernel='rbf',gamma = 0.1)

pca = PCA(n_components =1000 )#----------------,100,10,10000----------bayad set beshe
counetr =0
for train_index, test_index in kf:
    print counetr
    counetr = counetr+1
    X_train, X_test = SData[train_index], SData[test_index];
    y_train, y_test = labels[train_index], labels[test_index];
    #hal bayad kaheshe bod bedam
    X_trainNew = pca.fit_transform(X_train)
    X_testNew = pca.transform(X_test)
    
    #learn svm and calculate error
    RBFSVM.fit(X_trainNew,y_train)
    yValPredict2 = RBFSVM.predict(X_testNew)
    accScore = accScore+accuracy_score(y_test, yValPredict2)
    
accScore = accScore/3.0

