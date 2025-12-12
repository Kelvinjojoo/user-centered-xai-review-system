# User-Centered XAI for Fake Review Detection and Sentiment Analysis

This repository contains the full implementation of the research titled  
**“User-Centered Interpretation Dashboard for Explainable AI in Fake Review Detection and Sentiment Analysis.”**

The project provides an end-to-end pipeline: from data preprocessing, labeling, modeling, feature engineering, to explainability and a user-centered interactive dashboard.

All scripts, notebooks, datasets (processed), and models used in the study are included here for transparency and reproducibility.

---

## 🔍 Overview

This project uses the **Yelp Open Dataset** and applies:

- Traditional Machine Learning models (XGBoost, SVM, Random Forest, CatBoost, LightGBM)  
- Text features: TF-IDF + Sublinear TF + N-gram  
- Behavioral features: daily spike, hourly spike, similarity score  
- XAI methods: **LIME** (local interpretation) & **SHAP** (global interpretation)  
- User-centered dashboard built using **Streamlit**, ensuring explanations are understandable to non-technical users  

---

## 📁 Repository Structure
```
📦 root/
│
├── 1. Pre-Processing/
│ └──  Pre_processing.ipynb
│
├── 2. Labeling/
│ ├── Check_label.ipynb
│ └── Labeling.ipynb
│
├── 3. Modeling/
│ ├── Finalization_data.ipynb
│ ├── Sentiment_Analysis_XGBoost.pkl
│ ├── FakeReal_XGBoost.pkl
│ ├── SublinearTF_FakeReal_.ipynb (SVM, RF, CatBoost, LightGBM, XGBoost)
│ ├── SublinearTF_Sentiment_.ipynb (SVM, RF, CatBoost, LightGBM, XGBoost)
│ └── Pickle model/
│ ├── Sentiment_Analysis_.pkl
│ └── SublinearTF_FakeReal_.pkl
│
├── 4. Merge dataset for getting text/
│ ├── Merge.ipynb
│
└──  5. Dashboard/
 ├── check.ipynb
 ├── dashboard_review_analysis.py
 ├── FakeReal_XGBoost.pkl
 └── Sentiment_Analysis_XGBoost.pkl
```
---

## 🧠 Machine Learning Models

Models trained:

- **XGBoost** → Best performer (Fake Review & Sentiment)
- SVM
- Random Forest
- LightGBM
- CatBoost

Saved models are available in:
/3. Modeling/
/5. Dashboard/

---

## 🧪 Explainable AI (XAI)

### 🔹 SHAP (Global)
- Shows feature importance across dataset  
- Behavioral features emerge as top indicators (spike scores & similarity)

### 🔹 LIME (Local)
- Highlights important words & behavior for an individual review  
- Simplified for dashboard visualization  
- Used to support user-centered interpretation  

---

## 🖥️ Running the Streamlit Dashboard

The interactive dashboard lets users browse reviews, see predictions, and understand simplified explanations.

Run with:
cd "5. Dashboard"
streamlit run dashboard_review_analysis.py

---

## ⚠️ Dataset Usage Disclaimer

This project uses the **Yelp Open Dataset** under the official Terms of Service.

To comply with restrictions:
- Raw Yelp JSON files are not included
- Dashboard uses **dummy examples**
- Processed datasets contain **derived features only**

---

## 📚 Citation

If you use this repository for research:

Kelvin Jonathan Yusach, William, Henry Lucky,
Rilo Chandra Pradana, Noviyanti Tri Maretta Sagala.
User-Centered Interpretation Dashboard for Explainable AI
in Fake Review Detection and Sentiment Analysis, 2025.

---

## 👤 Authors

- Kelvin Jonathan Yusach — Conceptualization, Methodology, Software  
- William — Conceptualization, Methodology, Software  
- Henry Lucky — Supervision  
- Rilo Chandra Pradana — Supervision  
- Noviyanti Tri Maretta Sagala — Validation  
