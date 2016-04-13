import numpy as np
import urllib
from sklearn import cross_validation
url="https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
raw_data=urllib.urlopen(url)
print "raw:",raw_data
d=np.genfromtxt(raw_data,delimiter=",",dtype='str')
#dataset=np.loadtxt(raw_data, delimiter=",")
print d
#print(d.shape)
X = d[:, 1:5]
y = d[:, 0]
print "x:" , X

#knn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=20)
knnresult=cross_val_score(knn,X,y,cv=20)
print("knn Mean result",knnresult.mean())

#bagging
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
baggingresult=cross_val_score(bagging,X,y,cv=10)
print ("bagging mean result:",baggingresult.mean())

#RF
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=10,criterion='entropy')
RFresult=cross_val_score(RF,X,y,cv=20)
#print ("RF result array:",RFresult)
print ("RF mean result:",RFresult.mean())

#adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
ada=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(ada, X, y,cv=20)
adaResult=scores.mean()
print ("Adaboost mean result",adaResult.mean())

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gradientClf = GradientBoostingClassifier(n_estimators=100, learning_rate=1,max_depth=1)
gradientResult=cross_val_score(gradientClf,X,y,cv=20)
print ("Gradient mean result",gradientResult.mean())

