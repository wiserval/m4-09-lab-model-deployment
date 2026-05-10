# Iris Species Classifier API

## Overview
This API provides a production-ready interface for a Random Forest model trained on the Iris dataset. It allows users to classify iris flowers into three species: Setosa, Versicolor, and Virginica.

## How to Run
1. Install dependencies: `pip install flask joblib numpy scikit-learn`
2. Ensure `model.joblib` and `target_names.joblib` are in the root directory.
3. Start the server: `python app.py`
4. The API will be available at `http://localhost:5000`

## API Specification

### 1. Health Check
* **URL:** `/health`
* **Method:** `GET`
* **Response:** `{"status": "healthy"}`

### 2. Single Prediction
* **URL:** `/predict`
* **Method:** `POST`
* **Request Body:** `{"features": [5.1, 3.5, 1.4, 0.2]}`
* **Response:** `{"predicted_class": "setosa", "probabilities": {...}}`

### 3. Batch Prediction
* **URL:** `/predict_batch`
* **Method:** `POST`
* **Request Body:** `{"samples": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4]]}`
* **Response:** `{"predictions": [{"sample_index": 0, "predicted_class": "setosa"}, ...]}`