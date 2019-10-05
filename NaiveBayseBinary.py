from sklearn.naive_bayes import GaussianNB
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

#linear svm feature selection
lsvc = LinearSVC(C=0.5, penalty="l1", dual=False).fit(xtrain, bin_train)
model = SelectFromModel(lsvc, prefit=True)
X_train_new = model.transform(xtrain)
X_val_new = model.transform(xval)
X_test_new = model.transform(xtest)

#linear regression feature selection:

lr = LogisticRegression(C=0.5, penalty="l1", dual=False).fit(xtrain, bin_train)
model2 = SelectFromModel(lr, prefit=True)
X_train_new2 = model2.transform(xtrain)
X_val_new2 = model2.transform(xval)
X_test_new2 = model2.transform(xtest)



gnb = GaussianNB()
y_pred = gnb.fit(X_train_new, bin_train).predict(X_val_new)
tmpAcc = accuracy_score(bin_val, y_pred)
tmpF = f1_score(bin_val, y_pred)

y_pred2 = gnb.fit(X_train_new2, bin_train).predict(X_val_new2)
tmpAcc2 = accuracy_score(bin_val, y_pred2)
tmpF2 = f1_score(bin_val, y_pred2)


y_pred3 = gnb.fit(xtrain, bin_train).predict(xval)
tmpAcc3 = accuracy_score(bin_val, y_pred3)
tmpF3 = f1_score(bin_val, y_pred3)


y_pred_test = gnb.fit(xtrain, bin_train).predict(xtest)

#write to file
import pickle
file = open('bin_test_naiv', 'w')
pickle.dump(y_pred_test, file)
file.close()

#to read the content of the file use command : 
f = open('bin_test_naiv', 'r')
yTestPredic = pickle.load(f)