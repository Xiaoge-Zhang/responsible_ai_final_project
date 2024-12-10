import pandas as pd
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction, GridSearchReduction

pd.set_option('display.max_columns', None)

X = pd.read_csv('../data/cdc_Data_X.csv', index_col=['race', 'sex'])
y = pd.read_csv('../data/cdc_Data_y.csv', index_col=['race', 'sex'])

# set the random seed
rnd_seed = 123456
model_used = 'SVM'
protected_attrs = ['race']
result = {}
output_dir = '../output/'

# start using different in-process techniques
np.random.seed(0) #need for reproducibility

# split the data
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=rnd_seed)

# solve using Logistic Regression
estimator = LinearSVC(dual="auto")
y_pred = estimator.fit(X_train, y_train).predict(X_test)
model_acc = accuracy_score(y_test, y_pred)
dir_sex_default = disparate_impact_ratio(y_test, y_pred, prot_attr='sex', priv_group=1)
dir_race_default = disparate_impact_ratio(y_test, y_pred, prot_attr='race', priv_group=2)
aoe_sex_default = average_odds_error(y_test, y_pred, prot_attr='sex', priv_group=1)
aoe_race_default = average_odds_error(y_test, y_pred, prot_attr='race', priv_group=2)
aoe_default = average_odds_error(y_test, y_pred, prot_attr=['race', 'sex'], priv_group=(2, 1))
print('model: ', model_used)
print('accuray score: ', model_acc)
print('disparate impact ratio regarding sex: ', dir_sex_default)
print('disparate impact ratio regarding race: ', dir_race_default)
print('average odds error regarding sex: ', aoe_sex_default)
print('average odds error regarding race: ', aoe_race_default)
print('average odds error: ', aoe_default)

result['default'] = [dir_sex_default, dir_race_default, aoe_sex_default, aoe_race_default, aoe_default]


print()
print('using attribute reweighting')
# mitigation using race as protected attribute
rew = ReweighingMeta(estimator=LinearSVC(dual="auto"),
                     reweigher=Reweighing(protected_attrs))

params = {'estimator__C': [1, 5, 10]}

clf = GridSearchCV(rew, param_grid=params, scoring='accuracy', cv=5)
clf.fit(X_train, y_train.values.ravel())
reweight_acc = clf.score(X_test, y_test.values.ravel())
y_pred_reweight = clf.predict(X_test)
dir_sex_reweight = disparate_impact_ratio(y_test, y_pred_reweight, prot_attr='sex', priv_group=1)
dir_race_reweight = disparate_impact_ratio(y_test, y_pred_reweight, prot_attr='race', priv_group=2)
aoe_sex_reweight = average_odds_error(y_test, y_pred_reweight, prot_attr='sex', priv_group=1)
aoe_race_reweight = average_odds_error(y_test, y_pred_reweight, prot_attr='race', priv_group=2)
aoe_reweight = average_odds_error(y_test, y_pred_reweight, prot_attr=['race', 'sex'], priv_group=(2, 1))

print('disparate impact ratio regarding sex after applying reweighting: ', dir_sex_reweight)
print('disparate impact ratio regarding race after applying reweighting: ', dir_race_reweight)
print('average odds error regarding sex after applying reweighting: ', aoe_sex_reweight)
print('average odds error regarding race after applying reweighting: ', aoe_race_reweight)
print('average odds error after applying reweighting: ', aoe_reweight)

result['reweight'] = [dir_sex_reweight, dir_race_reweight, aoe_sex_reweight, aoe_race_reweight, aoe_reweight]

# mitigation using exponential gradient reduction
print()
print('using exponential gradient reduction')

# create the model
egr = ExponentiatedGradientReduction(prot_attr=protected_attrs,
                                              estimator=LinearSVC(dual="auto"),
                                              constraints="EqualizedOdds",
                                              drop_prot_attr=False)
X_train_in_process = X_train.reset_index()
X_test_in_process = X_test.reset_index()
egr.fit(X_train_in_process, y_train.values.ravel())
egr_acc = egr.score(X_test_in_process, y_test.values.ravel())
y_pred_egr = egr.predict(X_test_in_process)

dir_sex_egr = disparate_impact_ratio(y_test, y_pred_egr, prot_attr='sex', priv_group=1)
dir_race_egr = disparate_impact_ratio(y_test, y_pred_egr, prot_attr='race', priv_group=2)
aoe_sex_egr = average_odds_error(y_test, y_pred_egr, prot_attr='sex', priv_group=1)
aoe_race_egr = average_odds_error(y_test, y_pred_egr, prot_attr='race', priv_group=2)
aoe_egr = average_odds_error(y_test, y_pred_egr, prot_attr=['race', 'sex'], priv_group=(2, 1))
print('disparate impact ratio regarding sex after applying Exponential Gradient Reduction: ', dir_sex_egr)
print('disparate impact ratio regarding race after applying Exponential Gradient Reduction: ', dir_race_egr)
print('average odds error regarding sex after applying Exponential Gradient Reduction: ', aoe_sex_egr)
print('average odds error regarding race after applying Exponential Gradient Reduction: ', aoe_race_egr)
print('average odds error after applying Exponential Gradient Reduction: ', aoe_egr)

result['egr'] = [dir_sex_egr, dir_race_egr, aoe_sex_egr, aoe_race_egr, aoe_egr]

# mitigation using Grid Search Reduction
print()
print('using GridSearchReduction')

# create the model
gsr = GridSearchReduction(prot_attr=protected_attrs,
                                      estimator=LinearSVC(dual="auto"),
                                      constraints="EqualizedOdds",
                                      grid_size=20,
                                      drop_prot_attr=False)

gsr.fit(X_train_in_process, y_train.values.ravel())
gsr_acc = gsr.score(X_test_in_process, y_test.values.ravel())
y_pred_gsr = gsr.predict(X_test_in_process)

dir_sex_gsr = disparate_impact_ratio(y_test, y_pred_gsr, prot_attr='sex', priv_group=1)
dir_race_gsr = disparate_impact_ratio(y_test, y_pred_gsr, prot_attr='race', priv_group=2)
aoe_sex_gsr = average_odds_error(y_test, y_pred_gsr, prot_attr='sex', priv_group=1)
aoe_race_gsr = average_odds_error(y_test, y_pred_gsr, prot_attr='race', priv_group=2)
aoe_gsr = average_odds_error(y_test, y_pred_gsr, prot_attr=['race', 'sex'], priv_group=(2, 1))
print('disparate impact ratio regarding sex after applying Exponential Gradient Reduction: ', dir_sex_gsr)
print('disparate impact ratio regarding race after applying Exponential Gradient Reduction: ', dir_race_gsr)
print('average odds error regarding sex after applying Exponential Gradient Reduction: ', aoe_sex_gsr)
print('average odds error regarding race after applying Exponential Gradient Reduction: ', aoe_race_gsr)
print('average odds error after applying Exponential Gradient Reduction: ', aoe_gsr)

result['gsr'] = [dir_sex_gsr, dir_race_gsr, aoe_sex_gsr, aoe_race_gsr, aoe_gsr]

# save the result
output_df = pd.DataFrame(result)
full_save_dir = output_dir + model_used + '_' + '_'.join(protected_attrs) + '.csv'
output_df.to_csv(full_save_dir, index=False)
