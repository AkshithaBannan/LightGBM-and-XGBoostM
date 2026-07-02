# Titanic Survival Prediction: LightGBM vs. XGBoost

This project builds and evaluates two state-of-the-art Gradient Boosted Decision Tree (GBDT) frameworks—**LightGBM** and **XGBoost**—to predict passenger survival on the Titanic. The implementation includes comprehensive data visualization, preprocessing, feature engineering, and comparative performance analysis to determine the better-performing model on structured tabular data.

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset Summary](#dataset-summary)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Performance Comparison](#performance-comparison)
- [Comparative Analysis](#comparative-analysis)
- [Key Takeaways](#key-takeaways)

---

## Overview

The objective of this project is to compare the predictive performance of **LightGBM (`LGBMClassifier`)** and **XGBoost (`XGBClassifier`)** on the classic **Titanic Survival Prediction** dataset. The workflow includes:

- Data loading and merging
- Exploratory Data Analysis (EDA)
- Missing value analysis
- Feature engineering and preprocessing
- Categorical feature encoding
- Model training
- Performance evaluation using multiple classification metrics

The project demonstrates how modern Gradient Boosted Decision Tree (GBDT) algorithms perform on structured datasets without extensive hyperparameter tuning.

---

## Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
```

### Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost

---

## Dataset Summary

The project combines both **Titanic_train.csv** and **Titanic_test.csv** before preprocessing to ensure consistent feature engineering across both datasets.

| Attribute | Value |
|-----------|------:|
| Total Records | **1,309** |
| Initial Features | **12** |
| Target Variable | **Survived** |
| Prediction Type | Binary Classification |

### Missing Values

| Feature | Missing Values |
|---------|---------------:|
| Age | 263 |
| Fare | 1 |
| Cabin | 1,014 |
| Embarked | 2 |

---

## Data Preprocessing Pipeline

The following preprocessing steps were applied before model training.

### 1. Missing Value Imputation

- Filled missing **Age** values using the **median**
- Filled the missing **Fare** value using the **median**
- Filled missing **Embarked** values using the **mode ("S")**

---

### 2. Feature Elimination

The following columns were removed:

- **Cabin** (over 77% missing values)
- **PassengerId**
- **Name**
- **Ticket**

These features either contained excessive missing values or provided minimal predictive value for the baseline models.

---

### 3. Categorical Encoding

#### Label Encoding

Applied to:

- **Sex**

Example:

```
male   → 0
female → 1
```

#### One-Hot Encoding

Applied to:

- **Embarked**

Generated features:

- Embarked_C
- Embarked_Q
- Embarked_S

---

### 4. Train-Test Split

After preprocessing:

- Combined dataset was separated back into training and testing sets using the presence of the **Survived** label.
- Training data was further split into:

- **80% Training**
- **20% Validation**

using:

```python
random_state = 42
```

---

## Performance Comparison

The models were evaluated on the validation dataset using standard classification metrics.

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|---------|
| **LightGBM** | **82.12%** | **78.38%** | **78.38%** | **78.38%** |
| **XGBoost** | **79.33%** | **74.67%** | **75.68%** | **75.17%** |

---

## Comparative Analysis

### Accuracy

LightGBM achieved an accuracy of **82.12%**, outperforming XGBoost's **79.33%** by approximately **2.8 percentage points**.

### Precision

Both models demonstrated strong precision, indicating a relatively low number of false-positive predictions. LightGBM maintained a slight advantage.

### Recall

LightGBM more effectively identified passengers who survived, resulting in a higher recall than XGBoost.

### F1 Score

The balanced F1 Score of **78.38%** highlights LightGBM's ability to maintain an effective trade-off between precision and recall, making it the stronger performer for this dataset.

---

## Key Takeaways

- Both **LightGBM** and **XGBoost** achieved strong predictive performance on structured tabular data.
- **LightGBM** consistently outperformed **XGBoost** across all evaluation metrics in this baseline implementation.
- Histogram-based optimization enables LightGBM to train efficiently while maintaining excellent predictive performance.
- Although LightGBM produced better baseline results, optimal model selection should also consider:
  - Hyperparameter tuning
  - Dataset size
  - Training time
  - Memory consumption
  - Deployment requirements

---

## Conclusion

This project demonstrates the effectiveness of modern Gradient Boosted Decision Tree (GBDT) algorithms for binary classification problems. While both models performed competitively, **LightGBM** achieved superior accuracy, precision, recall, and F1 score on the Titanic dataset, making it the preferred baseline model in this comparison. Future improvements could include hyperparameter optimization, feature selection, cross-validation, and ensemble techniques to further enhance predictive performance.
