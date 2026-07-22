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

## 🏆 Results Summary

### K-Fold Cross Validation (5-Fold) — Mean Accuracy Comparison

| Model | Mean Accuracy | Std Dev | ROC-AUC |
|---|---|---|---|
| Random Forest | 98.63% | 0.50% | 0.9992 |
| SVM | 97.43% | 0.47% | 0.9967 |
| KNN | 96.91% | 1.31% | 0.9890 |
| **XGBoost** | **98.89%** | 0.58% | **0.9993** |
| Decision Tree | 97.60% | 0.88% | 0.9768 |
| **Ensemble (Voting)** | **98.80%** | **0.32%** | 0.9991 |

### Single Train-Test Split Results (80:20)

| Model | Test Accuracy |
|---|---|
| Random Forest | 98.72% |
| SVM | 97.44% |
| KNN | 97.86% |
| XGBoost | 97.86% |
| Decision Tree | 98.28% |

> 🏅 **Best Performing Model:** XGBoost (highest mean accuracy & ROC-AUC), closely followed by the Ensemble Voting Classifier (most stable — lowest standard deviation).

---

## 🧠 Explainable AI (SHAP)

To ensure transparency and trust in the model's predictions (crucial in healthcare), **SHAP (SHapley Additive exPlanations)** was used:

### 🔹 Global Explanation
- `shap.summary_plot()` — shows overall feature importance and impact direction across all predictions
- `shap.summary_plot(..., plot_type='bar')` — ranks features by average absolute SHAP value

### 🔹 Local Explanation (Individual Prediction)
- `shap.plots.waterfall()` — explains a **single prediction** by showing how each feature pushed the prediction from the base value to the final output
- `shap.force_plot()` — interactive visualization of feature contributions for a specific patient record

### Example Insight (Sample Patient)
For a sample prediction (Predicted Class: **High Risk**, Probability: 1.0):

| Feature | SHAP Contribution |
|---|---|
| BMI | +0.11 |
| Preexisting Diabetes | +0.11 |
| Blood Sugar (BS) | +0.06 |
| Gestational Diabetes | +0.06 |
| Heart Rate | +0.04 |
| Mental Health | +0.04 |

This confirms that **BMI, Preexisting Diabetes, and Blood Sugar** are the most influential factors driving high-risk pregnancy predictions — aligning with known clinical risk factors.

---

## 🧰 Tech Stack / Libraries Used

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
Google Colab (Environment)

### Key Modules
```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import shap
```

---

## 📁 Project Structure

```
Maternal-High-Risk-Pregnancy-Prediction/
│
├── Dataset/
│   └── Dataset - Updated.csv
│
├── Predicting_High_Risk_Pregnancy_in_Maternal_Healthcare_V1_with_SHAP.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/maternal-high-risk-pregnancy-prediction.git
cd maternal-high-risk-pregnancy-prediction
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

### 4. Dataset Setup
Place the dataset file `Dataset - Updated.csv` inside a `Dataset/` folder, or update the `DATA_PATH` variable in the notebook:
```python
DATA_PATH = "Dataset/Dataset - Updated.csv"
```

> Note: The original notebook was built in **Google Colab** and mounts Google Drive. If running locally or on Jupyter, replace the Drive mounting cell with a local file path.

---

## ▶️ How to Run

1. Open the notebook `Predicting_High_Risk_Pregnancy_in_Maternal_Healthcare_V1_with_SHAP.ipynb` in **Jupyter Notebook / JupyterLab / Google Colab**.
2. Run cells sequentially in this order:
   - Import Libraries
   - Load Dataset
   - EDA
   - Data Preprocessing
   - Train-Test Split
   - Train each model (RF, SVM, KNN, XGBoost, Decision Tree)
   - Run K-Fold Cross Validation for each model
   - Run Ensemble Voting Classifier
   - Run SHAP explainability section
3. View generated plots: Confusion Matrices, ROC Curves, SHAP Summary & Waterfall plots.

---

## 💡 Key Insights

- **XGBoost** and the **Ensemble Voting Classifier** deliver the best and most stable performance for this dataset.
- **Preexisting Diabetes, BMI, and Blood Sugar levels** are the strongest predictors of high-risk pregnancy.
- SHAP explanations make the model **clinically interpretable**, which is essential for healthcare adoption and trust.
- Feature correlation analysis shows expected medical relationships (e.g., BP correlations, diabetes-BS correlation).

---

## 🚀 Future Scope

- Deploy the model as a **web application** (Flask/Streamlit) for real-time risk prediction
- Integrate **hyperparameter tuning** (GridSearchCV / Optuna) for further optimization
- Expand dataset with more diverse demographic and clinical data
- Add **LIME** as an additional explainability method for comparison
- Build a **clinical dashboard** integrating SHAP visualizations for doctors

---

## 👤 Author

**Project:** Predicting High-Risk Pregnancy in Maternal Healthcare with SHAP  
**Domain:** Healthcare Analytics / Machine Learning / Explainable AI

---

