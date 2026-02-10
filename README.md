# 🏥 Health Insurance Cost Prediction

A Machine Learning project that predicts individual medical insurance costs based on demographic and lifestyle factors. This project includes a complete pipeline from Exploratory Data Analysis (EDA) to a deployed Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📌 Overview

The goal of this project is to assist in determining insurance premiums by analyzing key features such as age, BMI, smoking status, and region. The model was trained using various regression algorithms, achieving an **R² score of ~84%**, providing highly accurate cost estimations.

## 🚀 Key Features

* **Comprehensive EDA:** Detailed analysis of feature distributions and correlations (e.g., the impact of smoking and BMI on charges).
* **Data Preprocessing:** Handled missing values, encoded categorical variables (Label Encoding), and scaled numerical features (Standard Scaling).
* **Model Comparison:** Trained and evaluated multiple models to find the best performer:
    * Linear Regression
    * Random Forest Regressor (Best Performer)
    * Support Vector Regressor (SVR)
    * XGBoost Regressor
* **Interactive Web App:** Built a user-friendly interface using **Streamlit** where users can input their details and get instant cost predictions.

## 🛠️ Technologies Used

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
* **Deployment:** Streamlit
* **Model Persistence:** Joblib

After hyperparameter tuning and cross-validation, the **Random Forest Regressor** outperformed other models.

| Model | R² Score |
|-------|----------|
| **Random Forest** | **0.84** |
| XGBoost | 0.83 |
| Polynomial Regression | 0.82 |
| Linear Regression | 0.75 |

