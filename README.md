# 📊 Employee Attrition Prediction App

An interactive **Machine Learning web application** built with **Streamlit** to predict whether an employee is likely to leave the organization (Attrition).

---

## 🚀 Project Overview

This application uses a **Logistic Regression model** trained on HR analytics data to:

* Predict employee attrition (Yes/No)
* Display probability scores for both outcomes
* Highlight the **top 2 most influential features** affecting the prediction

---

## 🎯 Features

* ✅ User-friendly **interactive dashboard**
* ✅ Real-time prediction with a single click
* ✅ Probability display (Attrition Yes & No)
* ✅ Feature importance visualization
* ✅ Clean and responsive UI using Streamlit
* ✅ Preprocessing pipeline integrated with model

---

## 🧠 Machine Learning Model

* Algorithm: **Logistic Regression**
* Preprocessing:

  * Feature transformation using `FunctionTransformer`
  * Encoding categorical variables
  * Dropping irrelevant columns
* Model saved using **Joblib**

---

## 🖥️ Tech Stack

* Python 🐍
* Streamlit 📊
* Scikit-learn 🤖
* Pandas 📁
* Matplotlib 📉

---

## 📂 Project Structure

```
├── app.py                     # Streamlit application
├── attrition_model.joblib     # Trained ML model
├── README.md                  # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd <your-folder>
```

### 2️⃣ Create virtual environment (optional but recommended)

```
python -m venv venv
venv\\Scripts\\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the app

```
streamlit run app.py
```

---

## 📸 App Preview

* Input employee details
* Click **Predict**
* View:

  * Attrition result
  * Probability scores
  * Top influencing features graph

---

## ⚠️ Important Notes

* Ensure `attrition_model.joblib` is in the same directory as `app.py`
* Model expects input data in the **same format as training**
* Custom preprocessing functions must be defined before loading the model

---

## 🔥 Future Enhancements

* Add **SHAP Explainability**
* Deploy on **Streamlit Cloud**
* Add **downloadable prediction report**
* Improve UI with advanced visualizations

---

## 🙌 Acknowledgment

This project is built as part of a **Machine Learning & Data Science learning journey**, focusing on real-world HR analytics use cases.

---
