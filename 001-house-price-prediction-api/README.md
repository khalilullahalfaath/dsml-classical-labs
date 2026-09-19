# House Price Prediction API

> Regression model to predict house prices based on property features, deployed as REST API

**Status:** Completed | **Duration:** 8 days | **Level:** Beginner

- [House Price Prediction API](#house-price-prediction-api)
  - [Project Brief](#project-brief)
  - [Learning Objectives](#learning-objectives)
  - [Tech Stack](#tech-stack)
  - [Dataset](#dataset)
  - [Core Requirements](#core-requirements)
  - [Deliverables](#deliverables)
  - [Bonus Challenges](#bonus-challenges)
  - [Project Structure](#project-structure)
  - [🚀 Getting Started](#-getting-started)
    - [1. Setup Environment](#1-setup-environment)
    - [2. Download Dataset](#2-download-dataset)
    - [3. Run EDA \& Training](#3-run-eda--training)
    - [4. Start API Server](#4-start-api-server)
    - [5. Test API](#5-test-api)
  - [📈 Results](#-results)
    - [Model Comparison](#model-comparison)
    - [Feature Importance (Top 5)](#feature-importance-top-5)
    - [API Performance](#api-performance)
  - [📊 Model Card](#-model-card)
    - [Model Information](#model-information)
    - [Intended Use](#intended-use)
    - [Training Data](#training-data)
    - [Performance Metrics](#performance-metrics)
    - [Limitations](#limitations)
    - [Ethical Considerations](#ethical-considerations)
  - [📚 References](#-references)
  - [📝 Next Steps](#-next-steps)
  - [📧 Contact](#-contact)


## Project Brief

**The Request:**
"Client mau prediksi harga rumah berdasarkan fitur seperti luas, kamar, lokasi. Bikin API-nya ya, biar mereka tinggal input data rumah terus keluar estimasi harganya."

**Objective:** Build a house price prediction API that returns estimated price from property features.

**Success Criteria:**
- R² > 0.75 on test set
- API response time < 200ms
- Input validation and error handling

**Out of scope:** 
- Real-time streaming segmentation 
- CRM integration.

## Learning Objectives

-  Linear and polynomial regression fundamentals
-  Feature scaling and categorical encoding techniques
-  Model evaluation metrics: RMSE, MAE, R²
-  REST API development with FastAPI
-  Model deployment and serving

## Tech Stack

- **Language:** Python 3.10
- **ML Libraries:** scikit-learn, XGBoost, pandas, numpy
- **API Framework:** FastAPI
- **Visualization:** matplotlib, seaborn
- **Tools:** Jupyter, Postman

## Dataset

**Primary:** [KC House Data](https://www.kaggle.com/datasets/shivachandel/kc-house-data)
- 21,613 rows
- 21 features (bedrooms, bathrooms, sqft_living, location, etc.)

**Alternative:** [Ames Housing](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

## Core Requirements

- [x] EDA with visualizations (distributions, correlations, scatter plots)
- [x] Data preprocessing: handle missing values, encode categoricals, scale features
- [x] Train 3+ models (Linear Regression, Random Forest, XGBoost) and compare
- [x] Model evaluation with RMSE, MAE, and R² on test set
- [x] REST API with /predict endpoint accepting house features
- [x] Separate code into modules (preprocessing, training, serving)

## Deliverables

- [x] Jupyter notebook with full EDA and model comparison
- [x] Trained model artifact (.pkl)
- [x] REST API with Swagger documentation
- [x] Postman collection for API testing
- [x] README with model card (metrics, features, limitations)

## Bonus Challenges

- [ ] Feature importance visualization with SHAP
- [ ] Model versioning with timestamp and metrics tracking
- [ ] Data drift detection on incoming prediction requests 
- [ ] Simple frontend with Streamlit
  
## Project Structure
```
01-house-price-prediction-api/
├── data/
│   ├── raw/
│   │   └── kc_house_data.csv
│   └── processed/
│       └── processed_data.csv
├── notebooks/
│   ├── eda.ipynb
│   └── preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── api.py
├── models/
│   ├── xgboost_model.pkl
│   └── scaler.pkl
├── tests/
│   └── test_api.py
├── postman/
│   └── house_price_api.json
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Setup Environment
```
cd 01-house-price-prediction-api
    
python -m venv venv
source venv/bin/activate
    
pip install -r requirements.txt
```
Windows users: use `venv\Scripts\activate`

### 2. Download Dataset

Download from Kaggle and place in `data/raw/kc_house_data.csv`

### 3. Run EDA & Training
```
jupyter notebook notebooks/01-eda.ipynb
    
python src/train.py
```

### 4. Start API Server

`uvicorn src.api:app --reload`

API available at: http://localhost:8000
Swagger docs at: http://localhost:8000/docs

### 5. Test API
```
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": 98001,
    "lat": 47.5,
    "long": -122.3,
    "sqft_living15": 1800,
    "sqft_lot15": 4500
  }'
```

Or import Postman collection from `postman/House_Price_Prediction.postman_collection.json`

## 📈 Results

### Model Comparison

| Model | RMSE | MAE | R² Score | 
|-------|------|-----|----------|
| Linear Regression | $170,909 | $98,753 | 0.8068 |
| Decisio Tree | $212,076 | $104,113 | 0.7025 |
| **Random Forest** | **$145,894** | **$72,021** | **0.8592** |
| XGBoost | $148,198 | $77,344 | 0.8547 |

**Best Model:** Random Forest

### Feature Importance (Top 5)

1. `grade` - 0.330546
2. `sqft_living` - 0.260098
3. `lat` (latitude) - 0.152415
4. `long` (longitude) - 0.063791
5. `yr_built` - 0.033660

### API Performance

- **Response Time:** 69.6 ms (0.069604 seconds) ✅

## 📊 Model Card

### Model Information
- **Model Type:** Random Forest Regressor
- **Version:** 1.0.0
- **Training Date:** April 28, 2026
- **Framework:** scikit-learn 1.2.0

### Intended Use
- **Primary Use:** Estimating house prices based on property features for buyers, sellers, and real estate agents.
- **Users:**  Homeowners, real estate professionals, data analysts.
- **Out of Scope:** 

### Training Data
- **Source:** King County House Sales (2014-2015)
- **Size:** 21,613 samples
- **Split:** 80% train, 20% test

### Performance Metrics
- **R² Score:** 0.8592
- **RMSE:** $145,894
- **MAE:** $72,021

### Limitations


### Ethical Considerations
- Model may perpetuate historical biases in housing prices
- Should not be sole factor in pricing decisions
- Requires regular retraining with recent data



## 📚 References

- [Kaggle: KC House Data](https://www.kaggle.com/datasets/shivachandel/kc-house-data)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SHAP for Model Interpretability](https://github.com/slundberg/shap)

## 📝 Next Steps

1. Implement data drift detection
2. Build Streamlit frontend for easier interaction
3. Add CI/CD pipeline for automated testing
4. Deploy to cloud (AWS/GCP)
5. Implement A/B testing for model versions

## 📧 Contact

Questions or feedback? Open an issue or reach out!

---

**Project Timeline:** May 1 - May 8, 2026 | **Total Hours:** ~35 hours