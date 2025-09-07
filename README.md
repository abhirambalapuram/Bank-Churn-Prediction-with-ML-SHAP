# Bank-Churn-Prediction-with-ML-SHAP
End-to-end bank customer churn prediction using Python. Includes EDA, data cleaning, class imbalance handling, feature scaling, and multiple ML models (LogReg, RF, XGBoost) with cross-validation. Also explains predictions with SHAP.

**Overview**

This project builds an end-to-end pipeline to predict bank customer churn. It includes data cleaning, exploratory analysis, feature engineering, class-imbalance handling, model training/tuning, and explainability with SHAP to understand which features drive churn risk.

**Dataset**

Expected file: Churn_Modeling.csv (place it in data/ or update the path in the notebook).

Typical columns include demographics, account activity, and churn label.

**Methods**

Preprocessing: null/duplicate checks, outlier checks (LOF), scaling (StandardScaler).

Imbalance: RandomOverSampler (imbalanced-learn).

Models: Logistic Regression, Random Forest, XGBoost, SVM, KNN, Decision Tree, Gradient Boosting.

Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, 10-fold CV.

Explainability: SHAP summary/force/dependence plots.

**Requirements**

pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
matplotlib
seaborn
missingno


**How to Use**

Place Churn_Modeling.csv in data/ and adjust the path in the notebook if needed.

Run the notebook top-to-bottom.

Compare models via the metrics table and visualize feature importance with SHAP.

**Results You’ll See**

Cleaned dataset and EDA visuals.

Metrics (accuracy, precision, recall, F1, ROC-AUC) and confusion matrix.

SHAP plots highlighting which features most influence churn predictions.

**Notes**

SHAP visualizations can be compute-intensive; consider sampling for speed.

You can swap in your own dataset by matching column names and target label.
