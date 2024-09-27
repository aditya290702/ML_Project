import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

# Load data
Data = pd.read_csv("clinical_mastitis_cows.csv")
print(Data["class1"].value_counts())
print()

# Drop unnecessary columns
Data = Data.drop(["Cow_ID", "Breed", "House Number", "Address"], axis=1)

# Separate features (X) and target (Y)
X = Data.drop(["class1"], axis=1)
Y = Data["class1"]

# Initialize stratified k-fold cross-validator
kf = KFold(n_splits=10)

# Lists to store for logistic regression, XGBoost, and Random Forest
accuracy_scores = []
accuracy_scores_xgb = []
accuracy_scores_rfc = []

# Lists to store evaluation metrics for logistic regression, XGBoost, and Random Forest
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Standardize the features by scaling them
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train)
    X_test_scaled = std.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(acc)

    # Train XGBoost model
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_train_scaled, Y_train)
    Y_pred_xgb = model_xgb.predict(X_test_scaled)
    accuracy_xgb = accuracy_score(Y_test, Y_pred_xgb)
    accuracy_scores_xgb.append(accuracy_xgb)

    # Train Random Forest Regressor
    model_rfc = RandomForestClassifier()
    model_rfc.fit(X_train_scaled, Y_train)
    Y_pred_rfc = model_rfc.predict(X_test_scaled)
    accuracy_rfc = accuracy_score(Y_test, Y_pred_rfc)
    accuracy_scores_rfc.append(accuracy_rfc)

    # Evaluate models
    accuracy_scores.append(accuracy_score(Y_test, Y_pred_rfc))
    precision_scores.append(precision_score(Y_test, Y_pred_rfc))
    recall_scores.append(recall_score(Y_test, Y_pred_rfc))
    f1_scores.append(f1_score(Y_test, Y_pred_rfc))


# Calculate mean and standard deviation of r2 scores for logistic regression
mean_accuracy_score = np.mean(accuracy_scores)
std_accuracy_score = np.std(accuracy_scores)
print(f'Mean r2 Score for Logistic Regression: {mean_accuracy_score * 100:.2f}% And Std Deviation: {std_accuracy_score:.2f}')
print()

# Calculate mean and standard deviation of r2 scores for XGBoost
mean_accuracy_score_xgb = np.mean(accuracy_scores_xgb)
std_accuracy_score_xgb = np.std(accuracy_scores_xgb)
print(f'Mean r2 Score for XGBoost: {mean_accuracy_score_xgb * 100:.2f}% And Std Deviation: {std_accuracy_score_xgb:.2f}')
print()

# Calculate mean and standard deviation of r2 scores for Random Forest Classifier
mean_accuracy_score_rfc = np.mean(accuracy_scores_rfc)
std_accuracy_score_rfc = np.std(accuracy_scores_rfc)
print(f'Mean r2 Score for Random Forest: {mean_accuracy_score_rfc * 100:.2f}% And Std Deviation: {std_accuracy_score_rfc:.2f}')
print()




# Calculate mean and standard deviation of evaluation metrics for Random Forest Classifier
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f'Mean Accuracy : {mean_accuracy * 100} % And Std Deviation : {std_accuracy}')
print(f'Mean Precision : {mean_precision * 100} % And Std Deviation : {std_precision}')
print(f'Mean Recall : {mean_recall * 100} % And Std Deviation : {std_recall}')
print(f'Mean F1-score : {mean_f1 * 100} % And Std Deviation : {std_f1}')

