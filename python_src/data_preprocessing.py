import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference

data = pd.read_csv('../data/cdc_data.csv', index_col=['race', 'sex'])
x_columns = ['age_group', 'process', 'exposure_yn', 'symptom_status',
             'hosp_yn', 'icu_yn', 'underlying_conditions_yn']
y_column = 'death_yn'
X = data[x_columns]
y = data[y_column]

X.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
y.index = pd.MultiIndex.from_arrays(y.index.codes, names=y.index.names)
# Convert all columns to 'category' dtype
X = X.apply(lambda x: x.astype('category'))
# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array (DataFrame)
# Fit and transform the DataFrame
X_encoded = encoder.fit_transform(X)
# Convert the encoded result back into a DataFrame for readability
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(X.columns), index=X.index)
y = pd.Series(y.factorize(sort=True)[0], index=y.index)

X_encoded_df.to_csv('../data/cdc_Data_X.csv')
y.to_csv('../data/cdc_Data_y.csv')