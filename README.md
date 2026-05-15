# Medical Insurance Cost Prediction

This project focuses on predicting medical insurance charges using machine learning regression models on the Medical Insurance Cost Dataset. The workflow includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, and performance evaluation using multiple regression algorithms.

The objective of this project is to analyze the factors affecting insurance costs and build predictive models capable of estimating medical charges with high accuracy.

---

# Dataset Overview

The dataset contains demographic and health-related attributes of insurance beneficiaries.

| Feature    | Description               |
| ---------- | ------------------------- |
| `age`      | Age of the individual     |
| `sex`      | Gender                    |
| `bmi`      | Body Mass Index           |
| `children` | Number of dependents      |
| `smoker`   | Smoking status            |
| `region`   | Residential region        |
| `charges`  | Medical insurance charges |

---

# Project Workflow

## 1. Data Preprocessing

* Checked missing values
* Encoded categorical features
* Applied feature scaling
* Removed target skewness using log transformation

---

## 2. Feature Engineering

Additional features were created to improve model learning:

### Age Squared

Used to capture non-linear relationships between age and insurance charges.

```python id="4dyg6g"
df['age_squared'] = df['age'] ** 2
```

### BMI-Smoker Interaction Feature

Used to model the strong relationship between smoking status and BMI.

```python id="jlwm8i"
df['bmi_smoker'] = df['bmi'] * df['smoker_yes']
```

### Log Transformation of Target Variable

```python id="glzjlwm"
df['log_charges'] = np.log(df['charges'])
```

This reduced skewness and improved regression performance.

---

# Models Used

The following regression models were trained and evaluated:

* Linear Regression
* Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor

---

# Model Performance

| Model                   | Training R² | Testing R² |
| ----------------------- | ----------- | ---------- |
| Ridge Regression        | 0.77        | 0.86       |
| Decision Tree Regressor | 0.82        | 0.86       |
| Random Forest Regressor | 0.94        | 0.86       |
| XGBoost Regressor       | 0.87        | 0.86       |
| SVR                     | 0.45        | 0.86       |
| Linear Regression       | 0.82        | 0.81       |

---

# Key Observations

* Feature engineering significantly improved model performance.
* Log transformation reduced target variable skewness.
* The BMI-Smoker interaction feature contributed strongly to prediction quality.
* Random Forest and XGBoost achieved the highest overall performance.
* Decision Tree and Random Forest showed signs of overfitting due to high training scores.
* Ridge Regression demonstrated strong generalization capability with balanced train-test performance.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook / Kaggle Notebook

---


# Evaluation Metrics

Regression models were evaluated using:

* R² Score
* Training Score
* Testing Score

---

# Conclusion

This project demonstrates the complete machine learning workflow for regression analysis on structured healthcare cost data. Through preprocessing, feature engineering, and model comparison, the models achieved strong predictive performance, with XGBoost and Random Forest producing the best results.

The project highlights the importance of:

* proper categorical encoding,
* feature engineering,
* log transformation,
* and model regularization in improving regression performance.

---

# Future Improvements

Possible future enhancements include:

* Hyperparameter tuning
* Cross-validation
* Gradient Boosting optimization
* Deployment using Flask or Streamlit
* Model explainability using SHAP values

---

# Dataset Source

Dataset obtained from:

[Kaggle](https://www.kaggle.com)
