# 🚢 Titanic Survival Dashboard

An interactive machine learning dashboard that analyzes survival factors 
from the Titanic disaster and predicts individual survival probability.

🔴 **[Live Demo](https://titanic-dashboard-div6gjdgtts8rzstf29qnm.streamlit.app/)**

---

## Features

- Filter passengers by class and sex
- Visual survival analysis with bar charts
- Predict your own survival probability based on class, sex, and age group
- Trained Random Forest model with 80%+ accuracy

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Random Forest Classifier)
- Streamlit
- Seaborn, Matplotlib

## Run Locally

git clone https://github.com/Jambowana/titanic-dashboard.git
cd titanic-dashboard
pip install -r requirements.txt
streamlit run app.py

## Dataset

Kaggle Titanic Dataset — 891 training passengers, 418 test passengers.

## Model

Random Forest Classifier trained on engineered features:
Passenger Class, Sex, Age Group, Title, Fare Band, Embarked, SibSp, Parch
