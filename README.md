# ✈️ Flight Delay & Cancellation Risk Predictor

A machine learning application that predicts flight delays and cancellations using XGBoost, with a modern React frontend.

## Features

- **Real-time Predictions**: Get instant delay/cancellation risk scores
- **XGBoost Model**: Trained on 1M+ flight records
- **Interactive UI**: Modern, responsive React interface
- **RESTful API**: Flask backend with CORS support

## Tech Stack

- **Frontend**: React, Axios, Modern CSS
- **Backend**: Flask, XGBoost, Pandas, Scikit-learn
- **ML Model**: XGBoost Classifier

## Quick Start

### Backend Setup (with Virtual Environment)
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r ../requirements.txt

# Train the model
python train_model.py

# Start the API
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

Visit `http://localhost:3000` to use the app!

## Model Performance

The XGBoost model is trained on flight delay and cancellation data with features including:
- Departure time and date
- Airline and airport information
- Flight distance
- Historical delay patterns

## API Endpoints

- `POST /predict` - Get delay/cancellation prediction
- `GET /health` - Check API status
