import nfl_data_py as nfl #web scrapping for the combine data from 2000-2024
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn import preprocessing
import re
import numpy as np
from sklearn.datasets import load_iris
import pydot
import sklearn.svm as svm
import matplotlib.pyplot as plt
scraped_data = pd.DataFrame(nfl.import_combine_data([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]))
print(scraped_data)
scraped_data = scraped_data[['draft_year', 'draft_ovr', 'player_name','ht','wt','forty','bench','vertical','broad_jump','cone','shuttle']]
draft = [] # tell whether or not someone was drafted (target value)
print(scraped_data)
for row in scraped_data.itertuples(index=True):
   if (np.isnan(row.draft_ovr)):
        draft.append(False)
   else :
        draft.append(True)
scraped_data['drafted'] = draft
print(scraped_data)
X = scraped_data.iloc[:,5:11]
y = scraped_data.iloc[:,-1]
X.dropna(inplace=True)
y = y[y.index.isin(X.index)]
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
clf = DecisionTreeClassifier()
print(cross_val_score(clf, X_train, y_train, cv=7))
clf.fit(X_train, y_train)
print(metrics.accuracy_score(y_valid, clf.predict(X_valid)))
svm_cross = svm.SVC(kernel='linear', C=1)
print(cross_val_score(svm_cross,X_train, y_train, cv=7))
svm_cross.fit(X_train, y_train)
print(metrics.accuracy_score(y_valid, svm_cross.predict(X_valid)))
def sortSecond(val):
     return val[1]
values = clf.feature_importances_
features = list(X)
importances = [(features[i], values[i]) for i in range(len(features))]
importances.sort(reverse=True, key=sortSecond)
print(importances)
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.savefig('decisiontree.png')
plt.show()
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
 max_features=None, max_leaf_nodes=None,
 min_impurity_decrease=0.0,
 min_samples_leaf=1, min_samples_split=2,
 min_weight_fraction_leaf=0.0,
 random_state=None, splitter='best')
print(metrics.accuracy_score(y_valid, clf.predict(X_valid)))
params_dist = {
'criterion': ['gini', 'entropy'],
'max_depth': randint(low=4, high=40),
'max_leaf_nodes': randint(low=1000, high=20000),
'min_samples_leaf': randint(low=20, high=100),
'min_samples_split': randint(low=40, high=200)
}
clf_tuned = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(clf_tuned, params_dist, cv=7)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
best_tuned_clf = random_search.best_estimator_
print(metrics.accuracy_score(y_valid, best_tuned_clf.predict(X_valid)))
export_graphviz(
     best_tuned_clf,
     out_file='decisiontree.dot',
     feature_names=list(X_train),
     rounded=True,
     filled=True
)
# start of bagging code
clf = BaggingClassifier(random_state=0)
print(cross_val_score(clf, X_train, y_train, cv=10))
clf.fit(X_train, y_train)
print(metrics.accuracy_score(y_valid, clf.predict(X_valid)))
params_dist = {
     'n_estimators':randint(low=1,high=100),
     'max_samples':[x / 10 for x in range(1, 11)],
     'max_features':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.92, 0.95, 1.0]
}
#bagging classifer code starts here
clf_tuned = BaggingClassifier(random_state=42)
random_search = RandomizedSearchCV(clf_tuned, params_dist, cv=7)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
best_tuned_clf = random_search.best_estimator_
print(metrics.accuracy_score(y_valid, best_tuned_clf.predict(X_valid)))
# randomforest booster code
clf = RandomForestClassifier()
print(cross_val_score(clf, X_train, y_train, cv=10))
clf.fit(X_train, y_train)
print(metrics.accuracy_score(y_valid, clf.predict(X_valid)))
params_dist = {
'criterion': ['gini', 'entropy','log_loss'],
'max_depth': randint(low=4, high=40),
'max_leaf_nodes': randint(low=1000, high=20000),
'min_samples_leaf': randint(low=20, high=100),
'min_samples_split': randint(low=40, high=200)
}
clf_tuned = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(clf_tuned, params_dist, cv=7)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)
best_tuned_clf = random_search.best_estimator_
print(metrics.accuracy_score(y_valid, best_tuned_clf.predict(X_valid)))
export_graphviz(
     best_tuned_clf.estimators_[0],
     out_file='randomforesttree.dot',
     feature_names=list(X_train),
     rounded=True,
     filled=True
)