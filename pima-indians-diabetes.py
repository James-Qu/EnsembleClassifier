import numpy as np
import urllib
from sklearn import cross_validation
url1="https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data=urllib.urlopen(url1)
dataset=np.loadtxt(raw_data, delimiter=",")
#print(dataset.shape)
X = dataset[:, 0:8]
y = dataset[:, 8]
print X
#K-fold
"""from sklearn.cross_validation import KFold
kf=KFold(len(X),10)
print ("kf:",kf)
for train_index, test_index in kf:
    print("TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]"""

#split dataset
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data,train_target, test_size=0.4, random_state=0)
"""from sklearn.cross_validation import  train_test_split
train, test = train_test_split(dataset, test_size = 0.2)
trainx=train[:,0:7]
trainy=train[:,8]
testx=test[:,0:7]
testy=test[:,8]"""


"""from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_) """


#knn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=15,weights='distance')
knnresult=cross_val_score(knn,X,y,cv=10)
print("knn Mean result",knnresult.mean())

#bagging
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
baggingresult=cross_val_score(bagging,X,y,cv=10)
print ("bagging mean result:",baggingresult.mean())

#RF
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=10,criterion='entropy')
RFresult=cross_val_score(RF,X,y,cv=10)
#print ("RF result array:",RFresult)
print ("RF mean result:",RFresult.mean())

#adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
ada=AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(ada, X, y,cv=10)
adaResult=scores.mean()
print ("Adaboost mean result",adaResult.mean())

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gradientClf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
gradientResult=cross_val_score(gradientClf,X,y,cv=10)
print ("Gradient mean result",gradientResult.mean())