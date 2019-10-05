from sklearn import svm
import numpy
from scipy.sparse import csr_matrix
import h5py
import numpy
import sys
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

#abad az ghesmate ghabl peyda shode and

#seperation of classes : 
file = h5py.File('/Users/Tina/Documents/csr_ml2.h5', 'r')
data = file['data']
indices = file ['indices']
indptr = file['indptr']
shape = file['shape']
labels = file['labels'].value
SData = csr_matrix((data, indices, indptr), shape=shape).toarray()

numOfDim = 1000 ;

class4Indexes = (labels ==4)
class3Indexes = (labels ==3)
class2Indexes = (labels ==2)
class1Indexes = (labels ==1)
class0Indexes = (labels ==0)
x4 = SData[class4Indexes]
y4 = labels[class4Indexes]

x3 = SData[class3Indexes]
y3 = labels[class3Indexes]

x2 = SData[class2Indexes]
y2 = labels[class2Indexes]

x1 = SData[class1Indexes]
y1 = labels[class1Indexes]

x0 = SData[class0Indexes]
y0 =labels[class0Indexes]



#find numOfDim/numClasses pcs from each class
dim = numOfDim/5;
pca0 = PCA(n_components =dim )
pca0.fit_transform(x0)

pca1 = PCA(n_components =dim )
pca1.fit_transform(x1)

pca2 = PCA(n_components =dim )
pca2.fit_transform(x2)

pca3 = PCA(n_components =dim )
pca3.fit_transform(x3)

pca4 = PCA(n_components =dim )
pca4.fit_transform(x4)


#cumulative them in one matrix and check if it is inversable
all1234 = numpy.concatenate((pca1.components_,pca2.components_,pca3.components_,pca4.components_),axis=0)
tmpMatrix = pca0.components_
for i in range(len(all1234)):
    print i
    tmp = [all1234[i]];
    tmpMatrix = numpy.concatenate((tmpMatrix,tmp),axis=0)
    rank = numpy.linalg.matrix_rank(tmpMatrix)
    if(rank !=len(tmpMatrix)):
        tmpMatrix = tmpMatrix[0:len(tmpMatrix)-1]
        


#reduce dim using new matrix


#learn svm classifier and fid accuracy
RBFSVM = svm.SVC(C=1,kernel='rbf',gamma = 0.1)
numData = len(SData)
kf = KFold(numData, n_folds=3)
for train_index, test_index in kf:
    print counetr
    counetr = counetr+1
    X_train, X_test = SData[train_index], SData[test_index];
    y_train, y_test = labels[train_index], labels[test_index];
    RBFSVM.fit(X_trainNew,y_train)
    yValPredict2 = RBFSVM.predict(X_testNew)
    accScore = accScore+accuracy_score(y_test, yValPredict2)    
accScore = accScore/3.0