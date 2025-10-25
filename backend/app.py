from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
CORS(app)

# Load model and preprocessing info
MODEL_PATH = 'models/flight_delay_model.pkl'
PREPROCESSING_PATH = 'models/preprocessing_info.json'

try:
    model = joblib.load(MODEL_PATH)
    with open(PREPROCESSING_PATH, 'r') as f:
        preprocessing_info = json.load(f)
    print("✓ Model and preprocessing info loaded successfully")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None
    preprocessing_info = None

# Airport coordinates for distance calculation (major US airports)
AIRPORT_COORDS = {
    'JFK': (40.6413, -73.7781), 'LAX': (33.9416, -118.4085), 'ORD': (41.9742, -87.9073),
    'DFW': (32.8998, -97.0403), 'DEN': (39.8561, -104.6737), 'ATL': (33.6407, -84.4277),
    'SFO': (37.6213, -122.3790), 'SEA': (47.4502, -122.3088), 'LAS': (36.0840, -115.1537),
    'MCO': (28.4312, -81.3081), 'EWR': (40.6895, -74.1745), 'BOS': (42.3656, -71.0096),
    'CLT': (35.2144, -80.9473), 'MIA': (25.7959, -80.2870), 'PHX': (33.4352, -112.0101),
    'IAH': (29.9902, -95.3368), 'MSP': (44.8848, -93.2223), 'DTW': (42.2162, -83.3554),
    'PHL': (39.8729, -75.2437), 'LGA': (40.7769, -73.8740), 'BWI': (39.1774, -76.6684),
    'SLC': (40.7899, -111.9791), 'DCA': (38.8521, -77.0377), 'SAN': (32.7338, -117.1933),
    'TPA': (27.9755, -82.5332), 'PDX': (45.5898, -122.5951), 'HNL': (21.3187, -157.9225),
    'BNA': (36.1245, -86.6782), 'AUS': (30.1945, -97.6699), 'MDW': (41.7868, -87.7522),
    'DAL': (32.8470, -96.8517), 'OAK': (37.7126, -122.2197), 'SJC': (37.3639, -121.9289),
    'RDU': (35.8801, -78.7880), 'MSY': (29.9934, -90.2580), 'SAT': (29.5337, -98.4698),
    'RSW': (26.5361, -81.7550), 'PIT': (40.4915, -80.2329), 'CVG': (39.0469, -84.6678),
    'CMH': (39.9980, -82.8919), 'IND': (39.7173, -86.2944), 'MCI': (39.2976, -94.7139),
    'BUF': (42.9405, -78.7322), 'JAX': (30.4941, -81.6879), 'BDL': (41.9389, -72.6832),
    'ONT': (34.0560, -117.6012), 'BUR': (34.1977, -118.3587), 'SMF': (38.6954, -121.5901),
    'SNA': (33.6762, -117.8682), 'ALB': (42.7483, -73.8017), 'ABQ': (35.0402, -106.6092)
}

def calculate_distance(origin, dest):
    """Calculate distance between two airports in miles using Haversine formula"""
    if origin not in AIRPORT_COORDS or dest not in AIRPORT_COORDS:
        return 1000  # Default distance
    
    lat1, lon1 = AIRPORT_COORDS[origin]
    lat2, lon2 = AIRPORT_COORDS[dest]
    
    # Haversine formula
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_miles = 3959 * c  # Earth radius in miles
    
    return round(distance_miles, 2)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict flight delay/cancellation risk
    
    Expected JSON input:
    {
        "origin": "JFK",
        "dest": "LAX",
        "date": "2024-12-25",
        "hour": 8,
        "minute": 30,
        "period": "AM"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Extract input data
        origin = str(data.get('origin', 'JFK'))
        dest = str(data.get('dest', 'LAX'))
        flight_date = str(data.get('date', '2024-06-15'))  # Format: YYYY-MM-DD
        hour = int(data.get('hour', 8))
        minute = int(data.get('minute', 0))
        period = str(data.get('period', 'AM'))
        
        # Parse date
        from datetime import datetime
        date_obj = datetime.strptime(flight_date, '%Y-%m-%d')
        month = date_obj.month
        day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
        day_of_month = date_obj.day
        
        # Convert 12-hour to 24-hour format
        if period == 'PM' and hour != 12:
            hour_24 = hour + 12
        elif period == 'AM' and hour == 12:
            hour_24 = 0
        else:
            hour_24 = hour
        
        hour_of_day = hour_24
        
        # Calculate distance between airports
        distance = calculate_distance(origin, dest)
        
        # Feature engineering
        # Time of day category
        if 0 <= hour_of_day < 6:
            time_of_day = 'late_night'
        elif 6 <= hour_of_day < 9:
            time_of_day = 'early_morning'
        elif 9 <= hour_of_day < 12:
            time_of_day = 'morning'
        elif 12 <= hour_of_day < 15:
            time_of_day = 'midday'
        elif 15 <= hour_of_day < 18:
            time_of_day = 'afternoon'
        elif 18 <= hour_of_day < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        # Distance category
        if distance < 500:
            distance_category = 'short'
        elif distance < 1500:
            distance_category = 'medium'
        else:
            distance_category = 'long'
        
        # Basic features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_winter = 1 if month in [12, 1, 2] else 0
        is_summer = 1 if month in [6, 7, 8] else 0
        
        # Holiday periods
        is_thanksgiving = 1 if (month == 11 and 20 <= day_of_month <= 28) else 0
        is_christmas_period = 1 if ((month == 12 and day_of_month >= 18) or (month == 1 and day_of_month <= 5)) else 0
        is_spring_break = 1 if (month == 3 and 10 <= day_of_month <= 25) else 0
        is_summer_vacation = 1 if ((month == 6 and day_of_month >= 15) or month == 7 or (month == 8 and day_of_month <= 15)) else 0
        
        # Group origin airport
        label_encoders = preprocessing_info['label_encoders']
        origin_classes = label_encoders.get('origin_grouped', [])
        if origin not in origin_classes:
            origin_grouped = 'OTHER'
        else:
            origin_grouped = origin
        
        # Create feature vector
        feature_columns = preprocessing_info['feature_columns']
        categorical_columns = preprocessing_info['categorical_columns']
        numerical_columns = preprocessing_info['numerical_columns']
        
        # Encode categorical features
        features = []
        
        categorical_values = {
            'origin_grouped': origin_grouped,
            'time_of_day': time_of_day,
            'distance_category': distance_category
        }
        
        for col in categorical_columns:
            value = categorical_values.get(col, 'unknown')
            classes = label_encoders.get(col, [])
            if value in classes:
                encoded_value = classes.index(value)
            else:
                encoded_value = 0
            features.append(float(encoded_value))
        
        # Numerical features
        numerical_values = {
            'Month': month,
            'DayOfWeek': day_of_week,
            'DayOfMonth': day_of_month,
            'hour_of_day': hour_of_day,
            'distance': distance,
            'is_weekend': is_weekend,
            'is_winter': is_winter,
            'is_summer': is_summer,
            'is_thanksgiving': is_thanksgiving,
            'is_christmas_period': is_christmas_period,
            'is_spring_break': is_spring_break,
            'is_summer_vacation': is_summer_vacation
        }
        
        # Apply scaling
        scaler_mean = preprocessing_info.get('scaler_mean', [])
        scaler_scale = preprocessing_info.get('scaler_scale', [])
        
        for i, col in enumerate(numerical_columns):
            value = numerical_values.get(col, 0)
            if scaler_mean and scaler_scale and i < len(scaler_mean):
                scaled_value = (value - scaler_mean[i]) / scaler_scale[i]
            else:
                scaled_value = value
            features.append(float(scaled_value))
        
        # Make prediction
        features_array = np.array([features])
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Calibrate predictions
        raw_risk = prediction_proba[1] * 100
        
        if raw_risk < 30:
            risk_score = raw_risk * 0.5
        elif raw_risk < 50:
            risk_score = 15 + (raw_risk - 30) * 0.75
        elif raw_risk < 70:
            risk_score = 30 + (raw_risk - 50) * 0.6
        else:
            risk_score = 42 + (raw_risk - 70) * 0.4
        
        risk_score = round(min(risk_score, 55), 2)
        
        prediction = 1 if risk_score > 40 else 0
        
        # Determine risk level
        if risk_score < 15:
            risk_level = "Low"
            recommendation = "Your flight has a low risk of delay or cancellation."
        elif risk_score < 30:
            risk_level = "Medium"
            recommendation = "There's a moderate risk. Consider arriving early and having backup plans."
        else:
            risk_level = "High"
            recommendation = "Elevated risk detected. Check flight status frequently and consider alternatives."
        
        return jsonify({
            'prediction': prediction,
            'riskScore': risk_score,
            'riskLevel': risk_level,
            'recommendation': recommendation,
            'willBeDelayed': bool(prediction),
            'calculatedDistance': distance
        })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)