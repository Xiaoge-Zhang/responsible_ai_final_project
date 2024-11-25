import pandas as pd
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error

pd.set_option('display.max_columns', None)

X = pd.read_csv('../data/cdc_Data_X.csv', index_col=['race', 'sex'])
y = pd.read_csv('../data/cdc_Data_y.csv', index_col=['race', 'sex'])

# split the training data
# set the random seed
rnd_seed = 123456
model_used = 'SVM'
# Redirect stdout to a file
original_stdout = sys.stdout  # Save the original stdout
file = open("../output/{}_output.txt".format(model_used), "w")  # Open the file for writing
sys.stdout = file

# split the data
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=rnd_seed)

# solve using Logistic Regression
model = LinearSVC(dual="auto")
y_pred = model.fit(X_train, y_train).predict(X_test)
print('model: ', model_used)
print('accuray score: ', accuracy_score(y_test, y_pred))
print('disparate impact ratio regarding sex: ', disparate_impact_ratio(y_test, y_pred, prot_attr='sex', priv_group=1))
print('disparate impact ratio regarding race: ', disparate_impact_ratio(y_test, y_pred, prot_attr='race', priv_group=2))
print('average odds error: ', average_odds_error(y_test, y_pred, prot_attr=['race', 'sex'], priv_group=(2, 1)))

# mitigation using race as protected attribute
rew = ReweighingMeta(estimator=LinearSVC(dual="auto"),
                     reweigher=Reweighing('race'))

params = {'estimator__C': [1, 5, 10]}

clf = GridSearchCV(rew, param_grid=params, scoring='accuracy', cv=5)
clf.fit(X_train, y_train.values.ravel())
print('accuracy after applying reweighting', clf.score(X_test, y_test.values.ravel()))

# performance after reweighting
improved_result = disparate_impact_ratio(y_test, clf.predict(X_test), prot_attr='race')
print('disparate impact ratio regarding race after reweighting: ', improved_result)

improved_odds_error = average_odds_error(y_test, clf.predict(X_test), prot_attr=['race', 'sex'], priv_group=(2, 1))
print('average odds error regarding race  and sex after reweighting: ', improved_odds_error)

# close the file writer
# Reset stdout to the original stdout
sys.stdout = original_stdout
file.close()  # Close the file