---
layout: page
title: Credit Card Fraud Detection
description: A system to capture fraudulant credit card tranasactions
img: assets/img/proj_credit_card_fraud/1.png
importance: 2
category: work
related_publications: false
---

📂 [GitHub Repo Link](https://github.com/AjaydevNarathmeethal/Credit-Card-Fraud-Detection/)

# 💳 Credit Card Fraud Detection

- This project aims to detect fraudulent credit card transactions using machine learning techniques.  
- The dataset used is the **Credit Card Fraud Detection Dataset** available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
- The goal is to build a highly sensitive model that can accurately detect fraud cases while minimizing false negatives.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/FraudDetectionModel.gif" title="Demo of Site" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Demo of Site
</div>


---

## 🧠 Project Overview

- **Objective:** Identify fraudulent credit card transactions.
- **Dataset:** `creditcard.csv` (from Kaggle)
- **Tech Stack:**
  - **Languages & Tools:** Python, Jupyter Notebook, Gradio
  - **Libraries:** pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, imbalanced-learn
- **Machine Learning Approach:**
  - Exploratory Data Analysis (EDA)
  - Data preprocessing and handling of imbalanced data using **SMOTE**
  - Model training and evaluation with multiple algorithms
  - **Best model selected based on ROC-AUC and Recall**
  - Model deployment using **Gradio**

---

## 📁 Project Structure

```
CreditCard-Fraud-Detection/
│
├── fraud_detection_analysis.ipynb # Jupyter notebook for data analysis and model building
├── fraud_detection_model.py # Deployed model code using Gradio
├── best_model.pkl # Saved best performing model
├── requirements.txt # List of dependencies
└── README.md # Project documentation

```

---

## 🔍 Data Exploration

- Dataset contains **283,726 transactions** and **31 columns** (`Time`, `V1`–`V28`, `Amount`, and `Class`).
- Highly **imbalanced dataset**:
  - `Class = 0` → Non-fraudulent (99.83%)
  - `Class = 1` → Fraudulent (0.17%)

### 🧾 Sample Data Preview
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/df.head.png" title="Data Preview" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Preview of data
</div>


### 🔢 Class Distribution
<div class="row">
    <div class="col-sm-6">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/Class_distribution.png" title="Class Distribution" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6"> </div>
    <div class="col-sm-6 d-flex justify-content-center">Class Distribution</div>
</div>

---

## ⚙️ Data Preprocessing

- Standardized features using `StandardScaler`
- Split dataset into training and testing sets (80:20)
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes in the training data.

---

## 🤖 Model Building and Evaluation

Tested multiple models to find the best performing one based on **ROC-AUC** and **Recall** scores.

### Models Evaluated:
1. **XGBoost (with SMOTE)**
2. **XGBoost (with scale_pos_weight)**
3. **Isolation Forest**

---

### 🔹 XGBoost (with scale_pos_weight) — *Best Model*

| Metric | Value |
|--------|--------|
| **ROC-AUC** | 0.98341 |
| **Recall (Fraud Class)** |  0.90 |
| **Precision (Fraud Class)** | 0.23 |
| **F1-Score (Fraud Class)** | 0.36 |


> ✅ **Reason for selection:** High **Recall** and **ROC-AUC** — critical for fraud detection problems.

📸 ROC Curve comparison
<div class="row">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/ROC_curve_comparison.png" title="ROC curve comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<br>

📸 Confusion Matrix of best model 
<div class="row">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/Best_model_report.png" title="Best model report" class="img-fluid rounded z-depth-1" %}
    </div>
</div>



---

### 🔹 Isolation Forest (for anomaly detection)

| Metric | Value |
|--------|--------|
| **ROC-AUC** | 0.9603 |
| **Recall (Fraud Class)** | 0.89 |
| **Precision (Fraud Class)** | 0.03 |

While good at detecting anomalies, it performed poorly in precision compared to XGBoost.

---

## 💾 Model Saving

The **best performing XGBoost model** was saved as:
```python
joblib.dump(xgb_test, "best_model.pkl")
```


---

## 🌐 Model Deployment (Gradio App)

The model is deployed using **Gradio**, allowing users to input transaction features and get predictions in real time.

### 🧩 Deployment Script
File: `fraud_detection_model.py`

```python
iface = gr.Interface(
    fn=predict_model,
    inputs=gr.Textbox(label="Enter 30 features separated by commas"),
    outputs=gr.HTML(label="Result"),
    title="Fraud Detection Model",
    description="Enter transaction features to get prediction from the trained model."
)

iface.launch()
```

## 📈 Results Summary

<div class="container-fluid">
  <div class="table-responsive">
    <div class="row font-weight-bold flex-nowrap">
      <div class="col">Model</div>
      <div class="col">ROC-AUC</div>
      <div class="col">Recall</div>
      <div class="col">Precision</div>
      <div class="col">F1-Score</div>
      <div class="col">Notes</div>
    </div>
    <div class="w-100 my-2"></div>
    <div class="row flex-nowrap">
      <div class="col">XGBoost (SMOTE)</div>
      <div class="col">0.961</div>
      <div class="col">0.89</div>
      <div class="col">0.17</div>
      <div class="col">0.28</div>
      <div class="col">Good recall, balanced</div>
    </div>
    <div class="w-100 my-2"></div>
    <div class="row flex-nowrap font-weight-bold">
      <div class="col">XGBoost (balanced)</div>
      <div class="col"><strong>0.9834</strong></div>
      <div class="col"><strong>0.90</strong></div>
      <div class="col">0.23</div>
      <div class="col"><strong>0.36</strong></div>
      <div class="col">✅<strong>Best model</strong></div>
    </div>
    <div class="w-100 my-2"></div>
    <div class="row flex-nowrap ">
      <div class="col">Isolation Forest</div>
      <div class="col">0.960</div>
      <div class="col">0.89</div>
      <div class="col">0.03</div>
      <div class="col">0.06</div>
      <div class="col">High recall, low precision</div>
    </div>
  </div>
</div>



<br>

## 🧩 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Example requirements.txt:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
gradio
```

## 🚀 How to Run the Project
1. Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2. Run the Jupyter Notebook (for analysis)
```bash
jupyter notebook fraud_detection_analysis.ipynb
```
3. Run the Gradio App (for deployment)
```pytthon
python fraud_detection_model.py
```

Open the local URL generated by Gradio to interact with the model.

## Deployed model demo

<div class="row">
    <div class="col-sm-6 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/Fraud_Detected.png" title="Fraud Detected" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 mx-auto mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/proj_credit_card_fraud/No_Fraud.png" title="No Fraud Detected" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-6 d-flex justify-content-center">Fraud Detected</div>
    <div class="col-sm-6 d-flex justify-content-center">No Fraud Detected</div>
</div>





## 📊 Future Improvements

- Integrate deep learning models (e.g., Autoencoders)
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Add real-time transaction data monitoring