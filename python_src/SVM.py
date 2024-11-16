from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', None)

# Define features (X) and target (y)
base_dir = '../data/'
x_columns = ['case_month', 'res_state', 'age_group', 'sex', 'process', 'exposure_yn', 'symptom_status',
             'hosp_yn', 'icu_yn', 'underlying_conditions_yn']
y_column = 'death_yn'
races = ['American_Indian_Alaska_Native', 'Asian', 'Black', 'Multiple_Other', 'White']
accuracies = []
precisions = []
recalls = []
au_rocs = []

for race in races:
    data = pd.read_csv(base_dir + "cdc_data_{}.csv".format(race))
    # Split dataset into training set and test set
    X = data[x_columns]
    y = data[y_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109) # 70% training and 30% tes

    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf', probability=True) # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Get predicted probabilities for the positive class
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Compute the performance
    au_rocs.append(roc_auc_score(y_test, y_prob))
    accuracies.append(metrics.accuracy_score(y_test, y_pred))
    precisions.append(metrics.precision_score(y_test, y_pred))
    recalls.append(metrics.recall_score(y_test, y_pred))

result = pd.DataFrame({
    'race': races,
    'au_roc': au_rocs,
    'accuracy': accuracies,
    'precision': precisions,
    'recall': recalls,
})

result.to_csv('../output/svm_results.csv', index=False)