# Titanic Survival Prediction: LightGBM vs. XGBoost

This project builds and evaluates two state-of-the-art Gradient Boosted Decision Tree (GBDT) frameworks—**LightGBM** and **XGBoost**—to predict passenger survival on the Titanic. The implementation includes comprehensive data visualization, preprocessing, a complete executable script, and a comparative performance analysis.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset Summary](#dataset-summary)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Complete Implementation Code](#complete-implementation-code)
- [Performance Comparison](#performance-comparison)
- [Key Takeaways](#key-takeaways)

---

## Overview
The goal of this assignment is to compare the classification capabilities of LightGBM (`LGBMClassifier`) and XGBoost (`XGBClassifier`) using the classic Titanic dataset. The workflow tracks data ingestion, exploratory data analysis (EDA), structural handling of missing data, categorical feature encoding, and multi-metric model evaluations.

---

## Dependencies
The code is built using Python 3 and leverages standard data science and machine learning libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost

---

**## Dataset Summary**
The project processes both the training (Titanic_train.csv) and testing (Titanic_test.csv) sets combined for structured preprocessing:
Total Combined Rows: 1,309
Total Initial Features: 12
Target Variable: Survived (Binary: 0 = No, 1 = Yes)

Missing Value Inspection
Prior to preprocessing, missing values were identified across the combined dataset:
Age: 263 missing values
Fare: 1 missing value
Cabin: 1,014 missing values (Dropped due to sparsity)
Embarked: 2 missing values

Data Preprocessing Pipeline
To guarantee clean data feeding into the boosting algorithms, the following preprocessing steps are executed:
Imputation:
Missing Age values are filled using the dataset's median.
The single missing Fare value is filled using the median.
Missing Embarked locations are filled using the mode ('S').
Feature Elimination:
Dropped the Cabin column due to an excess of null parameters.
Dropped non-numeric metadata columns post-split: PassengerId, Name, and Ticket.
Categorical Encoding:
Sex is processed via Label Encoding (converting strings to sequential integers).
Embarked is converted using One-Hot Encoding (pd.get_dummies), splitting into Embarked_C, Embarked_Q, and Embarked_S.
Data Splitting:
The data is split back into train and test sets based on the presence of the Survived label.
The training subset is split into an 80/20 train-test validation partition (random_state=42).

## Complete Implementation Code
Below is the complete, self-contained Python script containing the entire data engineering and modeling workflow:Pythonimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

# 1. Loading the two datasets training and testing
train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')

# 2. Concatenating the two datasets into one for preprocessing
data = pd.concat([train_data, test_data], sort=False)

# 3. Data Visualization
# Plotting histogram
train_data.hist(bins=20, figsize=(12,10))
plt.tight_layout()
plt.show()

# Plotting boxplot
plt.figure(figsize=(10,8))
sns.boxplot(data=train_data.drop(columns=["PassengerId","Name","Ticket","Cabin"]))
plt.title("BOXPLOT FOR NUMERICAL FEATURES")
plt.show()

# Pairplot
sns.pairplot(data=train_data.drop(columns=["PassengerId","Name","Ticket","Cabin"]), hue="Survived")
plt.show()

# Feature relationships via barplots
sns.barplot(x="Pclass", y="Survived", data=train_data)
plt.title("SURVIVED RATE BY PASSENGER CLASS")
plt.show()

sns.barplot(x="Sex", y="Survived", data=train_data)
plt.title("SURVIVED RATE BY SEX")
plt.show()

sns.barplot(x="Embarked", y="Survived", data=train_data)
plt.title("SURVIVED RATE BY EMBARKED")
plt.show()

# 4. Preprocessing and Imputation
data["Age"].fillna(data['Age'].median(), inplace=True)
data["Fare"].fillna(data['Fare'].median(), inplace=True)
data["Embarked"].fillna(data['Embarked'].mode()[0], inplace=True)

# Dropping cabin as it has too many null values
data.drop(columns=["Cabin"], inplace=True)

# Encoding categorical variables
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])
data = pd.get_dummies(data, columns=["Embarked"])

# Splitting the data back into testing and training data sets
train_data = data[data["Survived"].notnull()]
test_data = data[data["Survived"].isnull()]

# Dropping unwanted metadata columns
train_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True, errors='ignore')
test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], inplace=True, errors='ignore')

# 5. Splitting the preprocessed training data into Train and Validation sets
X = train_data.drop(columns=["Survived"])
y = train_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation metric helper function
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# 6. Building LightGBM Model
lgbm_model = lgb.LGBMClassifier()
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
lgbm_metrics = evaluate_model(y_test, lgbm_pred)

# 7. Building XGBoost Model
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_metrics = evaluate_model(y_test, xgb_pred)

# 8. Comparative Analysis Matrix
results = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost'],
    'Accuracy': [lgbm_metrics[0], xgb_metrics[0]],
    'Precision': [lgbm_metrics[1], xgb_metrics[1]],
    'Recall': [lgbm_metrics[2], xgb_metrics[2]],
    'F1 Score': [lgbm_metrics[3], xgb_metrics[3]]
})

print(results)

# 9. Visualizing results
results.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.show()
Performance ComparisonFollowing evaluation on the 20% validation split, the baseline algorithms produced the following metrics:ModelAccuracyPrecisionRecallF1 ScoreLightGBM82.12%78.38%78.38%78.38%XGBoost79.33%74.67%75.68%75.17%Brief Report & Comparative AnalysisAccuracy: LightGBM marginally outperforms XGBoost on this validation slice by ~2.8%.Precision: Both models display high precision constraints, indicating a remarkably low rate of false positives.Recall: LightGBM excels at accurately identifying true positives (survivors) compared to XGBoost.F1 Score: LightGBM preserves a perfectly symmetrical balance between precision and recall (78.38%), making it the stronger performer for this specific dataset configuration.Key TakeawaysHigh-End Performance: Both algorithms demonstrate highly competitive classification capabilities on tabular structures.Efficiency: LightGBM natively optimizes training loops with built-in histogram methods, showing robust performance out-of-the-box even on smaller datasets without fine-tuning.Practical Implications: While LightGBM won this baseline round, actual production choices should weigh specific dataset scaling constraints, hyperparameter tuning limits, and environmental computational efficiency.
