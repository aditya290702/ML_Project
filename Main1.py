import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm



Data = pd.read_csv("clinical_mastitis_cows.csv")

Data_Type = Data.dtypes
print(Data_Type)

# # Filter the data for only Jersey breed cows with no previous mastitis history
# Data = Data[(Data["Breed"] == "Jersey") & (Data["Previous_Mastits_status"] == 0)]

# Drop unnecessary columns from the data
Data = Data.drop(["Cow_ID", "Breed", "House Number", "Address","Previous_Mastits_status"], axis=1)

# Separate features (X) and target (Y)
X = Data.drop(["class1"], axis=1)
Y = Data["class1"]

EUFL = Data["EUFL"]
plt.plot(EUFL)
plt.show()

# Initialize stratified k-fold cross-validator
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Lists to store r2 scores for logistic regression and XGBoost
r2_scores = []
r2_scores_xgb = []
r2_scores_rfc = []

for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]


    #Standardize the features by scaling them
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train)
    X_test_scaled = std.transform(X_test)

    #Train logistic regression model
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    r2score = r2_score(Y_test, Y_pred)
    r2_scores.append(r2score)

    #Train XGBoost model
    model_xgb = XGBClassifier(random_state=42)
    model_xgb.fit(X_train_scaled, Y_train)
    Y_pred_xgb = model_xgb.predict(X_test_scaled)
    r2_xgb = r2_score(Y_test, Y_pred_xgb)
    r2_scores_xgb.append(r2_xgb)

    #Random_Forest_Regressor
    model_rfc = RandomForestRegressor()
    model_rfc.fit(X_train_scaled, Y_train)
    Y_pred_rfc = model_xgb.predict(X_test_scaled)
    r2_rfc = r2_score(Y_test, Y_pred_xgb)
    r2_scores_rfc.append(r2_xgb)


#To Calculate mean and standard deviation of r2 scores for logistic regression
mean_r2_score = np.mean(r2_scores)
std_r2_score = np.std(r2_scores)
print()
print(f'Mean r2 Score : {mean_r2_score * 100} % And Std Deviation : {std_r2_score}')
print()

#To Calculate mean and standard deviation of r2 scores for XGBoost
mean_r2_score_xgb = np.mean(r2_scores_xgb)
std_r2_score_xgb = np.std(r2_scores_xgb)
print(f'XGBoost Mean r2 Score : {mean_r2_score_xgb * 100} % And Std Deviation : {std_r2_score_xgb}')
print()

#To Calculate mean and standard deviation of r2 scores for Random_Forest_Regressor
mean_r2_score_rfc = np.mean(r2_scores_rfc)
std_r2_score_rfc = np.std(r2_scores_rfc)
print(f'Random Forest Mean r2 Score : {mean_r2_score_rfc * 100} % And Std Deviation : {std_r2_score_rfc}')





