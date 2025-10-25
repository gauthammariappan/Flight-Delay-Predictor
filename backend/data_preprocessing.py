import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess the flight data with better feature engineering
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Initial data shape: {df.shape}")
    
    # Create delay indicator from actual delay columns
    df['weather_delay'] = pd.to_numeric(df['weather_delay'], errors='coerce').fillna(0)
    df['late_aircraft_delay'] = pd.to_numeric(df['late_aircraft_delay'], errors='coerce').fillna(0)
    df['cancelled'] = pd.to_numeric(df['cancelled'], errors='coerce').fillna(0)
    
    # Total delay in minutes
    df['total_delay'] = df['weather_delay'] + df['late_aircraft_delay']
    
    # Create target: delayed if total_delay > 15 minutes OR cancelled
    df['IsDelayedOrCancelled'] = ((df['total_delay'] > 15) | (df['cancelled'] == 1)).astype(int)
    
    print(f"Delay rate: {df['IsDelayedOrCancelled'].mean():.2%}")
    
    # Drop if target is missing
    df = df.dropna(subset=['IsDelayedOrCancelled'])
    
    # FEATURE ENGINEERING
    
    # 1. Time-based features
    df['Month'] = pd.to_numeric(df['month'], errors='coerce')
    df['DayOfWeek'] = pd.to_numeric(df['day_of_week'], errors='coerce')
    df['DayOfMonth'] = pd.to_numeric(df['day_of_month'], errors='coerce')
    
    # 2. Hour of day from dep_time
    df['dep_time'] = pd.to_numeric(df['dep_time'], errors='coerce')
    df['hour_of_day'] = (df['dep_time'] // 100).clip(0, 23)
    
    # 3. Time of day categories
    def categorize_time(hour):
        if pd.isna(hour):
            return 'normal'
        if 0 <= hour < 6:
            return 'late_night'
        elif 6 <= hour < 9:
            return 'early_morning'
        elif 9 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 15:
            return 'midday'
        elif 15 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    df['time_of_day'] = df['hour_of_day'].apply(categorize_time)
    
    # 4. Weekend indicator
    df['is_weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # 5. HOLIDAY PERIOD FEATURES - High traffic = delays
    # Thanksgiving week (November 20-28)
    df['is_thanksgiving'] = ((df['Month'] == 11) & (df['DayOfMonth'] >= 20) & (df['DayOfMonth'] <= 28)).astype(int)
    
    # Christmas/New Year period (Dec 18 - Jan 5)
    df['is_christmas_period'] = (
        ((df['Month'] == 12) & (df['DayOfMonth'] >= 18)) |
        ((df['Month'] == 1) & (df['DayOfMonth'] <= 5))
    ).astype(int)
    
    # Spring break (March 10-25)
    df['is_spring_break'] = ((df['Month'] == 3) & (df['DayOfMonth'] >= 10) & (df['DayOfMonth'] <= 25)).astype(int)
    
    # Summer vacation peak (June 15 - Aug 15)
    df['is_summer_vacation'] = (
        ((df['Month'] == 6) & (df['DayOfMonth'] >= 15)) |
        (df['Month'] == 7) |
        ((df['Month'] == 8) & (df['DayOfMonth'] <= 15))
    ).astype(int)
    
    # 6. WEATHER SEASON FEATURES
    # Winter weather delays (Dec-Feb)
    df['is_winter'] = df['Month'].isin([12, 1, 2]).astype(int)
    
    # Summer thunderstorm season (June-Aug)
    df['is_summer'] = df['Month'].isin([6, 7, 8]).astype(int)
    
    # 7. Distance and categories
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    def categorize_distance(dist):
        if pd.isna(dist):
            return 'unknown'
        if dist < 500:
            return 'short'
        elif dist < 1500:
            return 'medium'
        else:
            return 'long'
    
    df['distance_category'] = df['distance'].apply(categorize_distance)
    
    # 8. Origin airport (top 30 only)
    top_airports = df['origin'].value_counts().head(30).index
    df['origin_grouped'] = df['origin'].where(df['origin'].isin(top_airports), 'OTHER')
    
    # Select features for modeling
    categorical_columns = ['origin_grouped', 'time_of_day', 'distance_category']
    numerical_columns = [
        'Month', 'DayOfWeek', 'DayOfMonth', 'hour_of_day', 'distance', 
        'is_weekend', 'is_winter', 'is_summer',
        'is_thanksgiving', 'is_christmas_period', 'is_spring_break', 'is_summer_vacation'
    ]
    
    feature_columns = categorical_columns + numerical_columns
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        df[col] = df[col].astype(str).fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Handle numerical columns
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Drop any remaining NaN values
    df = df.dropna(subset=feature_columns + ['IsDelayedOrCancelled'])
    
    # Prepare final dataset
    X = df[feature_columns]
    y = df['IsDelayedOrCancelled']
    
    print(f"\nFeatures used: {feature_columns}")
    print(f"Total samples: {len(X)}")
    print(f"Delayed/Cancelled: {y.sum()} ({y.mean():.2%})")
    print(f"On-time: {(y == 0).sum()} ({(y == 0).mean():.2%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_columns': feature_columns,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'label_encoders': {col: list(label_encoders[col].classes_) for col in categorical_columns},
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    return X_train, X_test, y_train, y_test, preprocessing_info

def save_preprocessing_info(preprocessing_info, path='models/preprocessing_info.json'):
    """Save preprocessing information"""
    with open(path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    print(f"Preprocessing info saved to {path}")