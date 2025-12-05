# 🕵️‍♀️ Fake Job Posting Classifier

A deep-learning powered system that detects **fake vs. real job postings** using text fields such as job title, description, requirements, benefits, and company profile.
Built using **TensorFlow**, **LSTM networks**, and a **Streamlit-based web app**.

---

## 🚀 Project Overview

Recruitment fraud has grown significantly, making automated detection essential.
This project builds a **binary classification model** that flags fraudulent job postings by learning patterns across multiple text fields.

Key features:

* Combines **multiple datasets** to enhance data diversity
* Performs extensive **text preprocessing**
* Uses an **LSTM-based neural network** for classification
* Includes **threshold tuning** for optimal F1-score
* Exposes a clean **Streamlit UI** for real-time prediction

---

## 📂 Dataset

Two different datasets were merged to create a diverse training corpus:

### **Dataset Fields**

```
title  
company_profile  
description  
requirements  
benefits  
salary  
label (0 = real, 1 = fake)
```

### **Preprocessing Steps**

* Standardized all columns across datasets
* Filled missing optional fields with empty strings
* Removed rows missing **title** or **description** (mandatory inputs)
* Cleaned text (lowercasing, trimming, basic normalization)
* Converted salary to numeric
* Combined datasets using pandas and shuffled

---

## 🧠 Model Architecture

A deep learning pipeline using **TensorFlow**:

### **🔹 Text Vectorization**

Each text field is:

* Tokenized
* Converted to sequences
* Padded to a fixed length

### **🔹 LSTM Classifier**

The final model uses:

* Embedding layers
* LSTM layers (non-bidirectional performed best)
* Dense layers with dropout
* Sigmoid output for binary classification

Metrics evaluated:

* Accuracy
* Precision
* Recall
* F1-score
* Threshold-adjusted performance

---

## 📊 Model Performance

After training with early stopping and tuning:

* **Accuracy:** ~0.74
* **Precision:** ~0.69
* **Recall:** ~0.92
* **Best Threshold:** ~0.4568
* **F1 Score:** ~0.79

The model strongly identifies fake postings with high recall, while keeping false positives low via threshold selection.

---

## 🖥️ Streamlit Web App

A user-friendly interface where users can enter job details and instantly view the prediction.

### **Inputs**

* Job Title (**required**)
* Job Description (**required**)
* Company Profile
* Requirements
* Benefits

Optional fields default to empty strings.

### **Features**

* Clean, centered UI
* Styled buttons and layout
* Displays prediction (Real / Fake)
* Automatically formats missing fields
* Works with the saved trained model

---

## 🔧 Future Improvements

* Add BERT or RoBERTa for improved NLP performance
* Introduce explainability (LIME/SHAP)
* Add model confidence scores in UI
* Deploy using Docker or Cloud

---

## 📬 Contact

If you’d like help improving the model or deploying the app, feel free to reach out!
