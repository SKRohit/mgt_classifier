# all imports

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pickle import dump
from sklearn.feature_selection import RFECV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import *

# read training data
final_df = pd.read_csv("all_data/prepared_beemo.csv")
X = final_df.drop(["label","text"], axis=1)
y = final_df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


fhalf_scorer = make_scorer(fbeta_score, beta=0.5)
min_features_to_select = 1
clf = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring=fhalf_scorer,
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X_train_scaled, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {X_train.columns[rfecv.support_].values}")

# Select the features
X_rfe = X_train_scaled[:, rfecv.support_] 
# Train the final model using the selected features
clf.fit(X_rfe, y_train)
y_test_pred = clf.predict(X_test_scaled[:, rfecv.support_])
fscore = fbeta_score(y_test, y_test_pred, beta=0.5)
print(f"\nF1 Score on Test Set: {fscore}")

# save classifier and list of selected features
print("\n############# Saving Model #############")
with open("all_data/mgt_classifier.pkl", "wb") as f:
    dump(clf, f, protocol=5)
with open("all_data/features_list.pkl", "wb") as f:
    dump([rfecv.support_,X.columns[rfecv.support_].values], f, protocol=5)


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AI Generated", "Human Generated"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

