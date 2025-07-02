import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_trained_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('pipeline_anomaly_model.pkl')
        scaler = joblib.load('pipeline_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run anomaly_prediction_model.py first!")
        return None, None

def calculate_derived_features(flow_rate, diameter, length, roughness):
    """Calculate all derived features for prediction"""
    # Physical constants
    density = 1000  # kg/m³
    viscosity = 0.001  # Pa·s
    gravity = 9.81  # m/s²
    
    # Basic calculations
    area = np.pi * (diameter/2)**2
    velocity = flow_rate / area
    reynolds = density * velocity * diameter / viscosity
    
    # Friction factor (simplified Colebrook-White)
    relative_roughness = roughness / diameter
    if reynolds < 2300:
        friction_factor = 64 / reynolds
    else:
        friction_factor = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/(reynolds**0.9)))**2
    
    # Pressure calculations
    head_loss = friction_factor * (length/diameter) * (velocity**2/(2*gravity))
    pressure_drop = density * gravity * head_loss
    
    # Derived features
    pressure_gradient = pressure_drop / length
    flow_efficiency = 1.0  # Assume normal for new predictions
    reynolds_normalized = reynolds / 100000
    velocity_head = velocity**2 / (2 * gravity)
    anomaly_severity = 0.0  # Unknown for new data
    
    return [flow_rate, diameter, length, roughness, velocity, reynolds, 
            pressure_drop, pressure_gradient, flow_efficiency, reynolds_normalized,
            friction_factor, velocity_head, anomaly_severity]

def test_model_variations():
    """Test model with various input variations"""
    print("🧪 TESTING MODEL WITH VARIOUS INPUT VARIATIONS")
    print("="*60)
    
    model, scaler = load_trained_model()
    if model is None:
        return
    
    # Test scenarios with different conditions
    test_scenarios = [
        # [flow_rate, diameter, length, roughness, expected_anomaly]
        [1.5, 0.3, 1000, 0.00005, "Normal"],
        [0.8, 0.15, 1000, 0.00005, "Possible Blockage"],  # Small diameter, low flow
        [2.0, 0.3, 2000, 0.0008, "Corrosion"],  # High roughness
        [0.5, 0.3, 1000, 0.00005, "Possible Leak"],  # Very low flow
        [1.0, 0.1, 1000, 0.00005, "Major Blockage"],  # Very small diameter
        [1.2, 0.4, 500, 0.00005, "Normal"],  # Large diameter, short length
        [0.3, 0.2, 3000, 0.00015, "Multiple Issues"],  # Low flow + long pipe + rough
    ]
    
    results = []
    
    for i, (flow, dia, length, rough, expected) in enumerate(test_scenarios, 1):
        # Calculate features
        features = calculate_derived_features(flow, dia, length, rough)
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        # Get all class probabilities
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        results.append({
            'Test': i,
            'Flow Rate': flow,
            'Diameter': dia,
            'Length': length,
            'Roughness': rough,
            'Expected': expected,
            'Predicted': prediction,
            'Confidence': confidence,
            'Probabilities': prob_dict
        })
        
        print(f"\n📋 Test {i}: {expected}")
        print(f"   Input: Flow={flow}, Dia={dia}, Len={length}, Rough={rough}")
        print(f"   🎯 Predicted: {prediction} (confidence: {confidence:.3f})")
        print(f"   📊 Top 3 probabilities:")
        
        # Show top 3 predictions
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        for j, (cls, prob) in enumerate(sorted_probs[:3]):
            print(f"      {j+1}. {cls}: {prob:.3f}")
    
    return results

def stress_test_model():
    """Stress test with extreme values"""
    print("\n\n⚡ STRESS TESTING WITH EXTREME VALUES")
    print("="*60)
    
    model, scaler = load_trained_model()
    if model is None:
        return
    
    extreme_tests = [
        [0.01, 0.05, 100, 0.00001, "Extreme Low Flow"],
        [5.0, 0.8, 5000, 0.001, "Extreme High Values"],
        [1.0, 0.02, 1000, 0.00005, "Very Small Pipe"],
        [1.0, 1.0, 10000, 0.01, "Very Large Rough Pipe"],
    ]
    
    for i, (flow, dia, length, rough, description) in enumerate(extreme_tests, 1):
        try:
            features = calculate_derived_features(flow, dia, length, rough)
            features_scaled = scaler.transform([features])
            
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0].max()
            
            print(f"\n🔬 Extreme Test {i}: {description}")
            print(f"   Input: Flow={flow}, Dia={dia}, Len={length}, Rough={rough}")
            print(f"   🎯 Predicted: {prediction} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def batch_test_random_samples():
    """Test with random samples to check consistency"""
    print("\n\n🎲 BATCH TESTING WITH RANDOM SAMPLES")
    print("="*60)
    
    model, scaler = load_trained_model()
    if model is None:
        return
    
    # Generate 20 random test samples
    np.random.seed(42)
    predictions = []
    
    for i in range(20):
        # Random parameters within realistic ranges
        flow = np.random.uniform(0.1, 2.0)
        diameter = np.random.uniform(0.1, 0.5)
        length = np.random.uniform(500, 3000)
        roughness = np.random.uniform(0.00005, 0.0003)
        
        features = calculate_derived_features(flow, diameter, length, roughness)
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max()
        
        predictions.append(prediction)
        
        if i < 5:  # Show first 5 results
            print(f"Sample {i+1}: {prediction} (conf: {confidence:.3f})")
    
    # Show distribution of predictions
    pred_counts = pd.Series(predictions).value_counts()
    print(f"\n📊 Prediction Distribution from 20 random samples:")
    for anomaly, count in pred_counts.items():
        print(f"   {anomaly}: {count} samples ({count/20*100:.1f}%)")

def performance_benchmark():
    """Benchmark model performance on test data"""
    print("\n\n⏱  PERFORMANCE BENCHMARK")
    print("="*60)
    
    model, scaler = load_trained_model()
    if model is None:
        return
    
    # Load test data
    try:
        df = pd.read_csv('pipeline_anomaly_data.csv')
        X = df.drop('anomaly', axis=1)
        y = df['anomaly']
        
        # Use last 300 samples as test set
        X_test = X.tail(300)
        y_test = y.tail(300)
        
        X_test_scaled = scaler.transform(X_test)
        
        # Time the predictions
        import time
        start_time = time.time()
        predictions = model.predict(X_test_scaled)
        end_time = time.time()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        print(f"✅ Test Set Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   Prediction Time: {(end_time - start_time)*1000:.2f} ms for 300 samples")
        print(f"   Average per sample: {(end_time - start_time)/300*1000:.2f} ms")
        
    except FileNotFoundError:
        print("❌ Test data not found. Please run dataset_generator.py first!")

if __name__ == "__main__":
    print("🚀 PIPELINE ANOMALY DETECTION - MODEL TESTING")
    print("="*60)
    print(f"👤 User: nischal2805")
    print(f"📅 Date: 2025-05-25 16:22:14 UTC")
    print("="*60)
    
    # Run all tests
    test_results = test_model_variations()
    stress_test_model()
    batch_test_random_samples()
    performance_benchmark()
    
    print("\n\n✅ ALL TESTS COMPLETED!")
    print("="*60)
    print("🎯 Model testing finished successfully!")
    print("📊 Check results above for model performance analysis")
    print("🚀 Ready to use Streamlit interface!")


