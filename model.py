import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def train_anomaly_detection_model():
    """Train and evaluate pipeline anomaly detection model"""
    
    print("Loading pipeline anomaly dataset...")
    # Load the generated dataset
    df = pd.read_csv('pipeline_anomaly_data.csv')
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'anomaly']
    X = df[feature_columns]
    y = df['anomaly']
    
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {y.unique()}")
    
    # Split the data (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*50)
    print("TRAINING ANOMALY DETECTION MODEL")
    print("="*50)
    
    # Train Random Forest model with optimized parameters
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Model trained successfully!")
    print(f"✓ Test Accuracy: {accuracy:.3f}")
    
    # Detailed classification report
    print("\n" + "="*50)
    print("DETAILED PERFORMANCE METRICS")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*50)
    print(feature_importance.head(10))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n✓ Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test predictions on new samples
    print("\n" + "="*50)
    print("TESTING ON NEW PIPELINE CONDITIONS")
    print("="*50)
    
    # Test case 1: Normal pipeline
    test_normal = [[1.2, 0.3, 1500, 0.00008, 16.98, 61128, 45000, 30, 1.0, 0.61, 0.022, 14.7, 0.0]]
    test_normal_scaled = scaler.transform(test_normal)
    pred_normal = model.predict(test_normal_scaled)[0]
    prob_normal = model.predict_proba(test_normal_scaled)[0].max()
    print(f"Test 1 - Normal Pipeline: {pred_normal} (confidence: {prob_normal:.3f})")
    
    # Test case 2: Major blockage
    test_blockage = [[0.8, 0.3, 1500, 0.00008, 11.31, 40752, 85000, 56.7, 0.67, 0.41, 0.028, 6.5, 0.8]]
    test_blockage_scaled = scaler.transform(test_blockage)
    pred_blockage = model.predict(test_blockage_scaled)[0]
    prob_blockage = model.predict_proba(test_blockage_scaled)[0].max()
    print(f"Test 2 - Major Blockage: {pred_blockage} (confidence: {prob_blockage:.3f})")
    
    # Test case 3: Large leak
    test_leak = [[0.6, 0.3, 1500, 0.00008, 8.49, 30564, 25000, 16.7, 0.5, 0.31, 0.025, 3.7, 0.9]]
    test_leak_scaled = scaler.transform(test_leak)
    pred_leak = model.predict(test_leak_scaled)[0]
    prob_leak = model.predict_proba(test_leak_scaled)[0].max()
    print(f"Test 3 - Large Leak: {pred_leak} (confidence: {prob_leak:.3f})")
    
    # Test case 4: Corrosion
    test_corrosion = [[1.0, 0.3, 1500, 0.0006, 14.15, 50940, 75000, 50, 1.0, 0.51, 0.035, 10.2, 0.6]]
    test_corrosion_scaled = scaler.transform(test_corrosion)
    pred_corrosion = model.predict(test_corrosion_scaled)[0]
    prob_corrosion = model.predict_proba(test_corrosion_scaled)[0].max()
    print(f"Test 4 - Corrosion: {pred_corrosion} (confidence: {prob_corrosion:.3f})")
    
    # Save the trained model and scaler for future use
    import joblib
    joblib.dump(model, 'pipeline_anomaly_model.pkl')
    joblib.dump(scaler, 'pipeline_scaler.pkl')
    print(f"\n✓ Model saved as 'pipeline_anomaly_model.pkl'")
    print(f"✓ Scaler saved as 'pipeline_scaler.pkl'")
    
    return model, scaler, accuracy, feature_importance

def predict_new_anomaly(flow_rate, diameter, length, roughness, model_path='pipeline_anomaly_model.pkl'):
    """Function to predict anomaly for new pipeline data"""
    import joblib
    
    # Load saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load('pipeline_scaler.pkl')
    
    # Calculate derived features (simplified)
    velocity = flow_rate / (3.14159 * (diameter/2)**2)
    reynolds = 1000 * velocity * diameter / 0.001
    pressure_drop = 0.025 * (length/diameter) * (velocity**2/2) * 1000
    
    # Create feature array (with estimated values for other features)
    features = [[flow_rate, diameter, length, roughness, velocity, reynolds, 
                pressure_drop, pressure_drop/length, 1.0, reynolds/100000, 
                0.025, velocity**2/(2*9.81), 0.0]]
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    confidence = model.predict_proba(features_scaled)[0].max()
    
    return prediction, confidence

if __name__ == "__main__":
    # Train the model
    model, scaler, accuracy, feature_importance = train_anomaly_detection_model()
    
    print("\n" + "="*60)
    print("🎯 PIPELINE ANOMALY DETECTION MODEL READY!")
    print("="*60)
    print(f"✓ Final Model Accuracy: {accuracy:.1%}")
    print("✓ Model can predict: Normal, Partial_Blockage, Major_Blockage, Small_Leak, Large_Leak, Corrosion")
    print("✓ Ready for anomaly detection on new pipeline data!")
    print("="*60)


