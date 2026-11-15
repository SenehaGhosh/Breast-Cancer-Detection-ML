# 🎗️ Breast Cancer Detection Using Machine Learning

An end-to-end Machine Learning classification web application built with **Scikit-learn**, **XGBoost**, and **Streamlit** to assist in predicting whether a breast mass is **Malignant** or **Benign** based on digitized Fine Needle Aspirate (FNA) cell nuclei characteristics.

---

## 📋 Overview
Early and accurate diagnosis of breast cancer is critical for effective treatment. This project benchmarks multiple supervised machine learning algorithms on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, optimizes model hyperparameters to minimize false negatives, and deploys the winning model as an interactive web interface.

---

## 📊 Dataset Details
- **Source:** Wisconsin Diagnostic Breast Cancer (WDBC) Dataset (via `sklearn.datasets.load_breast_cancer`)
- **Total Samples:** 569 instances
- **Features:** 30 real-valued numeric attributes describing the mean, standard error, and "worst" (largest) values of cell nuclei traits (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension)
- **Target Classes:**
  - `0` — **Malignant** (212 samples)
  - `1` — **Benign** (357 samples)

---

## ⚙️ Machine Learning Pipeline
1. **Stratified Train-Test Split:** 80% training / 20% testing split preserving class proportions.
2. **Feature Scaling:** `StandardScaler` fitted strictly on training data to prevent data leakage.
3. **Model Benchmarking:** Evaluates **Logistic Regression**, **Support Vector Machine (RBF Kernel)**, **Random Forest**, and **XGBoost** using ROC-AUC and Recall metrics.
4. **Hyperparameter Tuning:** Uses 5-fold cross-validation (`GridSearchCV`) to optimize the top-performing classifier.
5. **Model Serialization:** Saves the final model (`breast_cancer_model.pkl`) and scaler (`scaler.pkl`) via `joblib`.

---

## 🖥️ Web Application Features
- **Manual Feature Sliders:** Interactive controls to simulate tumor nuclei measurements and get instant diagnostic predictions with confidence probabilities.
- **Batch CSV Upload:** Upload a CSV file containing multiple patient records for bulk diagnostic inference.

---

## 📁 Project Structure
```text
├── train_model.py              # Model training, comparison, tuning & artifact export
├── app.py                      # Streamlit web application UI
├── breast_cancer_model.pkl     # Serialized best-performing ML classifier
├── scaler.pkl                  # Fitted StandardScaler object
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
