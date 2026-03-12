# Nutrition-Based Diabetes Risk Prediction (NHANES)

This project implements a Machine Learning pipeline to predict diabetes risk using nutritional and demographic data from the **NHANES (National Health and Nutrition Examination Survey)**. 

The primary goal is to provide an explainable screening tool that prioritizes **Recall**, ensuring that high-risk individuals are correctly identified for further medical testing.

## Key Results
The pipeline successfully identified that a **Logistic Regression** model with balanced class weights provides the best screening performance for this dataset.

| Model | Accuracy | Recall (Sensitivity) | AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **75.0%** | **82.5%** | **0.84** |
| **XGBoost** | 76.5% | 69.0% | 0.84 |
| **Random Forest** | 91.7% | 1.5% | 0.82 |

> **Key Insight:** While Random Forest achieved the highest accuracy, it failed as a screening tool (low Recall). **Logistic Regression** is the recommended model, capturing **82.5% of positive cases**.

## Explainable AI (XAI)
Using **XGBoost Feature Importance**, the project identified the most significant risk factors for diabetes in the studied population:

1. **Age (RIDAGEYR):** 59.8% importance (The dominant predictor).
2. **Total Sugars (DR1TSUGR):** 11.5% importance.
3. **Dietary Fiber (DR1TFIBE):** 10.1% importance.
4. **Total Calories (DR1TKCAL):** 9.7% importance.
5. **Gender (RIAGENDR):** 8.7% importance.

## Project Structure
```text
Proiect_ML_Nutritie/
├── data/
│   └── raw/           # Raw NHANES .xpt files
├── notebooks/         # Exploratory Data Analysis & Prototyping
├── src/               # Modular Python code
│   ├── data_loader.py # Data ingestion logic
│   ├── preprocessing.py # Cleaning and feature engineering
│   └── models_engine.py # Training and evaluation logic
├── main.py            # Main entry point to run the pipeline
├── requirements.txt   # Project dependencies
└── README.md          # Documentation
```
### Author
Developed by Diana - Data Science Project 2026.

