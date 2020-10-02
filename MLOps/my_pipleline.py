"""
A test to see if I can pickle a model as a pipeline and not have issues with a custom model warning
"""
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Read the train and test data
TRAIN_DATA_CLF = '../data/surgical_dataset_train.csv'
TEST_DATA_CLF = '../data/surgical_dataset_test.csv'

clf_X_train = pd.read_csv(TRAIN_DATA_CLF)
clf_Y_train = clf_X_train.pop('complication')

clf_X_test = pd.read_csv(TEST_DATA_CLF)

# Fit the model as a pipeline with an imputer
clf_rf_model = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                         ("forest", RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0))])
clf_rf_model.fit(clf_X_train, clf_Y_train)

# Pickle the file and write it to the file system
if not os.path.exists('custom_model_clf'):
    os.makedirs('custom_model_clf')
with open('custom_model_clf/clf_rf_model_pipeline.pkl', 'wb') as pkl:
    pickle.dump(clf_rf_model, pkl)

# Call predict to confirm it works
clf_rf_model.predict(clf_X_test)

threshold = 0.3
predicted_proba = clf_rf_model.predict_proba(clf_X_test)
predicted = (predicted_proba [:,1] >= threshold).astype('int')
predicted
# accuracy_score(clf_Y_test, predicted)
