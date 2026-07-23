# Diabetes Prediction System

## About Project

This project is a Diabetes Prediction System made using Machine Learning. It predicts whether a person is Diabetic or Non-Diabetic based on some medical details entered by the user.

The machine learning model is trained using the Diabetes Dataset and the K-Nearest Neighbors (KNN) algorithm. The application is developed using Gradio and deployed on Render.

## Input Details

The user has to enter:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Output

The application predicts:

- Diabetic
- Non-Diabetic

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Gradio

## Project Files

- diabetes.csv – Dataset
- train_model.py – Code used to train the model
- diabetes_model.pkl – Trained machine learning model
- app.py – Gradio application
- requirements.txt – Required libraries
- README.md – Project information

## How to Run

1. Install all required libraries.

```
pip install -r requirements.txt
```

2. Run the application.

```
python app.py
```

3. Open the generated link in the browser.

## Model Used

K-Nearest Neighbors (KNN)

## Deployment

The project is uploaded on GitHub and deployed using Render.