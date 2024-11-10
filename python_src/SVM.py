from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)

data = pd.read_csv('../data/processed_data.csv')
data = data.apply(LabelEncoder().fit_transform)

# Define features (X) and target (y)
X = data[['age', 'sex', 'country', 'symptoms', 'chronic_disease_binary', 'travel_history_binary']]  # Select features
y = data['outcome']                  # Select target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109) # 70% training and 30% tes

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

# Save test samples, true labels, and predictions to a DataFrame
results_df = pd.DataFrame(X_test, columns=['age', 'sex', 'country', 'symptoms',
                                           'chronic_disease_binary', 'travel_history_binary'])  # Include feature columns
results_df['true_label'] = y_test.values
results_df['predicted_label'] = y_pred

# Save the results to a CSV file
results_df.to_csv('../output/svm_test_results.csv', index=False)