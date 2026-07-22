# 🤰 Predicting High-Risk Pregnancy in Maternal Healthcare using Machine Learning & Explainable AI (SHAP)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-red.svg)

A machine learning project that predicts **high-risk pregnancies** in maternal healthcare using clinical and health parameters. The project implements multiple classification algorithms, compares their performance using K-Fold Cross Validation, builds an ensemble model, and uses **SHAP (SHapley Additive exPlanations)** for model interpretability — helping healthcare professionals understand *why* a prediction was made.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Data Preprocessing](#-data-preprocessing)
- [Models Implemented](#-models-implemented)
- [Model Evaluation Strategy](#-model-evaluation-strategy)
- [Results Summary](#-results-summary)
- [Explainable AI (SHAP)](#-explainable-ai-shap)
- [Tech Stack / Libraries Used](#-tech-stack--libraries-used)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Key Insights](#-key-insights)
- [Future Scope](#-future-scope)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

Maternal health risk prediction is a critical application of machine learning in healthcare. Identifying high-risk pregnancies early can help doctors take preventive action and reduce maternal and infant mortality. This project builds and compares **six machine learning models** and an **ensemble voting classifier** to classify pregnancies as **High Risk** or **Low Risk**, based on vital health indicators such as blood pressure, blood sugar, BMI, heart rate, and pre-existing medical conditions.

To make the model's predictions trustworthy and interpretable for medical practitioners, **SHAP (Explainable AI)** is integrated to provide both **global** (overall feature importance) and **local** (individual prediction) explanations.

---

## ❓ Problem Statement

Given a set of maternal health parameters, predict whether a pregnancy is classified as **High Risk** or **Low Risk**, enabling early medical intervention and better prenatal care planning.

**Target Variable:** `Risk Level` (Binary: High / Low)

---

## 📊 Dataset

- **File:** `Dataset - Updated.csv`
- **Shape:** 1205 rows × 12 columns (before cleaning)
- **Shape after cleaning:** 1166 rows × 12 columns

### Features

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Systolic BP | Systolic Blood Pressure |
| Diastolic | Diastolic Blood Pressure |
| BS | Blood Sugar Level |
| Body Temp | Body Temperature |
| BMI | Body Mass Index |
| Previous Complications | History of pregnancy complications (0/1) |
| Preexisting Diabetes | Diabetes before pregnancy (0/1) |
| Gestational Diabetes | Diabetes during pregnancy (0/1) |
| Mental Health | Mental health condition indicator (0/1) |
| Heart Rate | Heart rate (bpm) |
| **Risk Level (Target)** | High / Low |

### Missing Values Handling
Missing values were found in `Systolic BP`, `Diastolic`, `BS`, `BMI`, `Previous Complications`, `Preexisting Diabetes`, `Heart Rate`, and `Risk Level`. All rows with missing values were **dropped** using `df.dropna()`.

### Class Distribution
- **Low Risk:** ~702 samples
- **High Risk:** ~464 samples

---

## 🔄 Project Workflow

Data Loading → Data Cleaning → EDA → Preprocessing (Encoding + Scaling)
→ Feature Importance Analysis → Train-Test Split (80:20)
→ Model Training (6 Algorithms) → K-Fold Cross Validation (k=5)
→ Ensemble Voting Classifier → Model Evaluation
→ Explainable AI using SHAP (Global + Local Explanations)


---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset shape & structure (`.info()`, `.describe()`)
- Null value inspection (`.isnull().sum()`)
- **Target class distribution** plot (Countplot)
- **Histogram** of all numerical features
- **Correlation Heatmap** to study feature relationships

### Key Correlation Insights
- `Systolic BP` and `Diastolic` are highly correlated (**0.79**)
- `BS` correlates strongly with `Age` (**0.60**) and `Preexisting Diabetes` (**0.55**)
- `Mental Health` correlates with `Previous Complications` (**0.45**)

---

## 🛠 Data Preprocessing

1. **Target Encoding:** `LabelEncoder` applied to `Risk Level` (object → numeric)
2. **Categorical Encoding:** `LabelEncoder` applied to categorical columns
3. **Feature Scaling:** `StandardScaler` applied to numerical features
4. **Feature Importance Check:** A temporary `RandomForestClassifier` was trained to identify the top 10 most important features
5. **Train-Test Split:** 80% Training / 20% Testing (Stratified Split)
   - Training samples: 932
   - Testing samples: 234

### Top Important Features (Random Forest)
1. Preexisting Diabetes
2. BMI
3. BS (Blood Sugar)
4. Heart Rate
5. Mental Health
6. Gestational Diabetes
7. Previous Complications

---

## 🤖 Models Implemented

| # | Model | Algorithm |
|---|---|---|
| 1 | **RF** | Random Forest Classifier |
| 2 | **SVM** | Support Vector Machine (RBF Kernel) |
| 3 | **KNN** | K-Nearest Neighbors |
| 4 | **XGB** | XGBoost Classifier |
| 5 | **DT** | Decision Tree Classifier |
| 6 | **Ensemble** | Soft Voting Classifier (SVM + KNN + DT + XGBoost) |

Each model was evaluated using:
- Simple Train/Test Split (80:20)
- **5-Fold Stratified Cross-Validation**
- Confusion Matrix
- ROC Curve & AUC Score
- Classification Report (Precision, Recall, F1-score)

---

## 📈 Model Evaluation Strategy

- **Cross-Validation:** `StratifiedKFold` with `k=5`, `shuffle=True`, `random_state=42`
- **Metrics Used:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Ensemble Strategy:** Soft Voting with weighted contributions:
  ```python
  weights = [1, 1, 1, 2]  # SVM, KNN, DT, XGBoost
