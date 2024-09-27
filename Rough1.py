import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

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

# Lists to store evaluation metrics for logistic regression, XGBoost, and Random Forest
accuracy_scores_lr = []
accuracy_scores_xgb = []
accuracy_scores_rfc = []
precision_scores = []
recall_scores = []
f1_scores = []

# Placeholder for SHAP analysis
explainer_rf = None
explainer_xgb = None
shap_values_rf = []
shap_values_xgb = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Standardize the features by scaling them
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train)
    X_test_scaled = std.transform(X_test)

    # Train logistic regression model
    model_lr = LogisticRegression()
    model_lr.fit(X_train_scaled, Y_train)
    Y_pred_lr = model_lr.predict(X_test_scaled)
    accuracy_scores_lr.append(accuracy_score(Y_test, Y_pred_lr))

    # Train XGBoost model
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_train_scaled, Y_train)
    Y_pred_xgb = model_xgb.predict(X_test_scaled)
    accuracy_scores_xgb.append(accuracy_score(Y_test, Y_pred_xgb))

    # Train Random Forest Classifier
    model_rfc = RandomForestClassifier()
    model_rfc.fit(X_train_scaled, Y_train)
    Y_pred_rfc = model_rfc.predict(X_test_scaled)
    accuracy_scores_rfc.append(accuracy_score(Y_test, Y_pred_rfc))

    # Evaluate Random Forest Classifier
    precision_scores.append(precision_score(Y_test, Y_pred_rfc))
    recall_scores.append(recall_score(Y_test, Y_pred_rfc))
    f1_scores.append(f1_score(Y_test, Y_pred_rfc))

    # SHAP analysis for Random Forest
    if explainer_rf is None:
        explainer_rf = shap.Explainer(model_rfc, X_train)
    shap_values_rf.append(explainer_rf(X_test))

    # SHAP analysis for XGBoost
    if explainer_xgb is None:
        explainer_xgb = shap.Explainer(model_xgb, X_train)
    shap_values_xgb.append(explainer_xgb(X_test))

# Calculate mean and standard deviation of accuracy scores for Logistic Regression
mean_accuracy_lr = np.mean(accuracy_scores_lr)
std_accuracy_lr = np.std(accuracy_scores_lr)
print(f'Mean Accuracy for Logistic Regression: {mean_accuracy_lr * 100:.2f}% and Std Deviation: {std_accuracy_lr:.2f}')
print()

# Calculate mean and standard deviation of accuracy scores for XGBoost
mean_accuracy_xgb = np.mean(accuracy_scores_xgb)
std_accuracy_xgb = np.std(accuracy_scores_xgb)
print(f'Mean Accuracy for XGBoost: {mean_accuracy_xgb * 100:.2f}% and Std Deviation: {std_accuracy_xgb:.2f}')
print()

# Calculate mean and standard deviation of accuracy scores for Random Forest Classifier
mean_accuracy_rfc = np.mean(accuracy_scores_rfc)
std_accuracy_rfc = np.std(accuracy_scores_rfc)
print(f'Mean Accuracy for Random Forest: {mean_accuracy_rfc * 100:.2f}% and Std Deviation: {std_accuracy_rfc:.2f}')
print()

# Calculate mean and standard deviation of evaluation metrics for Random Forest Classifier
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f'Mean Precision: {mean_precision * 100:.2f}% and Std Deviation: {std_precision:.2f}')
print(f'Mean Recall: {mean_recall * 100:.2f}% and Std Deviation: {std_recall:.2f}')
print(f'Mean F1-score: {mean_f1 * 100:.2f}% and Std Deviation: {std_f1:.2f}')

# Concatenate SHAP values for all folds
shap_values_rf = np.concatenate([sv.values for sv in shap_values_rf], axis=0)
shap_values_xgb = np.concatenate([sv.values for sv in shap_values_xgb], axis=0)

# SHAP summary plots
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_rf, X, plot_type="bar", show=False)
plt.title("SHAP Summary Plot for Random Forest")
plt.tight_layout()
plt.savefig("shap_summary_rf.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_xgb, X, plot_type="bar", show=False)
plt.title("SHAP Summary Plot for XGBoost")
plt.tight_layout()
plt.savefig("shap_summary_xgb.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_rf, X, show=False)
plt.title("SHAP Beeswarm Plot for Random Forest")
plt.tight_layout()
plt.savefig("shap_beeswarm_rf.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_xgb, X, show=False)
plt.title("SHAP Beeswarm Plot for XGBoost")
plt.tight_layout()
plt.savefig("shap_beeswarm_xgb.png")
