# ❤️ CodeAlpha Disease Prediction from Medical Data

## 📌 Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship.

The objective is to predict the likelihood of heart disease using patient medical data and machine learning algorithms. The system analyzes health parameters and provides a risk prediction along with probability scores.

---

## 🎯 Objective

Predict whether a patient is likely to have heart disease based on medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- Number of Colored Vessels
- Thalassemia

---

## 📊 Dataset

**Dataset:** Heart Disease UCI Dataset

The dataset contains various patient health records and a target variable indicating the presence or absence of heart disease.

Target Values:

- 0 = No Heart Disease
- 1 = Heart Disease

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

Three classification algorithms were compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

---

## 📈 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 79.02%   |
| Decision Tree       | 98.54%   |
| Random Forest       | 98.54%   |

### Selected Model

✅ Random Forest Classifier

### Final Accuracy

**98.54%**

---

## 📋 Classification Report

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.99   |
| Recall    | 0.99   |
| F1-Score  | 0.99   |
| Accuracy  | 98.54% |

---

## 📊 Generated Visualizations

The project automatically generates:

- Confusion Matrix
- ROC Curve
- Feature Importance Graph
- Model Comparison Chart

All screenshots are available inside the **screenshots/** folder.

---

## 💻 Streamlit Web Application

The project includes a user-friendly Streamlit interface.

Features:

- Patient Data Input Form
- Disease Risk Prediction
- Risk Probability Percentage
- Low / Medium / High Risk Classification
- Interactive User Interface

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
CodeAlpha_DiseasePrediction
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data
│   └── HeartDiseaseTrain-Test.csv
│
├── model
│   ├── heart_model.pkl
│   └── encoders.pkl
│
├── screenshots
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── model_comparison.png
```

---

## 🔮 Future Improvements

- Deploy on Streamlit Cloud
- Add XGBoost Model
- Add Explainable AI (SHAP)
- Support Multiple Disease Prediction
- Improve UI/UX

---

## 👨‍💻 Author

**Sukhraj Singh**

Machine Learning Intern at CodeAlpha

---

## 📜 License

This project is created for educational and internship purposes under CodeAlpha Machine Learning Internship.
