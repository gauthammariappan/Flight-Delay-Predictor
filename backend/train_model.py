import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
import os
import sys
from data_preprocessing import load_and_preprocess_data, save_preprocessing_info

def train_xgboost_model(csv_path):
    """
    Train XGBoost model with HEAVILY constrained parameters
    """
    print("="*60)
    print("FLIGHT DELAY PREDICTOR - MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessing_info = load_and_preprocess_data(csv_path)
    
    # Save preprocessing info
    os.makedirs('models', exist_ok=True)
    save_preprocessing_info(preprocessing_info)
    
    # Calculate scale_pos_weight
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count
    
    print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}:1")
    print("Training XGBoost with REALISTIC weights...")
    
    # HEAVILY constrained parameters - prevent overfitting on time
    model = xgb.XGBClassifier(
        n_estimators=100,           # Fewer trees
        max_depth=3,                # VERY shallow trees
        learning_rate=0.01,         # VERY slow learning
        subsample=0.7,              # Less data per tree
        colsample_bytree=0.5,       # Sample HALF the features per tree
        colsample_bynode=0.5,       # Further feature sampling
        scale_pos_weight=scale_pos_weight * 0.7,  # Reduce false positives
        min_child_weight=10,        # Much higher - prevents small splits
        gamma=0.5,                  # Heavy pruning
        reg_alpha=0.5,              # Strong L1 regularization
        reg_lambda=2.0,             # Strong L2 regularization
        max_delta_step=1,           # Limit prediction changes
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=15
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['On-time', 'Delayed/Cancelled']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")
    
    # Save model
    model_path = 'models/flight_delay_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    feature_importance = sorted(
        zip(preprocessing_info['feature_columns'], model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"{i:2d}. {feat:25s}: {importance:.4f}")
    
    return model, preprocessing_info

if __name__ == "__main__":
    csv_path = "data/flight_delay_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)
    
    train_xgboost_model(csv_path)
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)