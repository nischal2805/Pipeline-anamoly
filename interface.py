import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pipeline Anomaly Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .anomaly-normal { color: #28a745; }
    .anomaly-warning { color: #ffc107; }
    .anomaly-danger { color: #dc3545; }
    .feature-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler (cached)"""
    try:
        model = joblib.load('pipeline_anomaly_model.pkl')
        scaler = joblib.load('pipeline_scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

def calculate_features(flow_rate, diameter, length, roughness, input_pressure=None):
    """Calculate all derived features"""
    density = 1000
    viscosity = 0.001
    gravity = 9.81
    
    area = np.pi * (diameter/2)**2
    velocity = flow_rate / area
    reynolds = density * velocity * diameter / viscosity
    
    relative_roughness = roughness / diameter
    if reynolds < 2300:
        friction_factor = 64 / reynolds
    else:
        friction_factor = 0.25 / (np.log10(relative_roughness/3.7 + 5.74/(reynolds**0.9)))**2
    
    # Use input pressure if provided, otherwise calculate from physics
    if input_pressure is not None:
        pressure_drop = input_pressure
    else:
        head_loss = friction_factor * (length/diameter) * (velocity**2/(2*gravity))
        pressure_drop = density * gravity * head_loss
    
    pressure_gradient = pressure_drop / length
    flow_efficiency = 1.0
    reynolds_normalized = reynolds / 100000
    velocity_head = velocity**2 / (2 * gravity)
    anomaly_severity = 0.0
    
    return [flow_rate, diameter, length, roughness, velocity, reynolds, 
            pressure_drop, pressure_gradient, flow_efficiency, reynolds_normalized,
            friction_factor, velocity_head, anomaly_severity]

def get_anomaly_color(anomaly):
    """Get color for anomaly type"""
    colors = {
        'Normal': '#28a745',
        'Partial_Blockage': '#ffc107', 
        'Major_Blockage': '#dc3545',
        'Small_Leak': '#fd7e14',
        'Large_Leak': '#dc3545',
        'Corrosion': '#6f42c1'
    }
    return colors.get(anomaly, '#6c757d')

def get_anomaly_description(anomaly):
    """Get description for anomaly type"""
    descriptions = {
        'Normal': "✅ Pipeline operating within normal parameters",
        'Partial_Blockage': "⚠ Partial obstruction detected - monitor closely",
        'Major_Blockage': "🚨 Major blockage detected - immediate attention required",
        'Small_Leak': "⚠ Minor leak detected - schedule maintenance",
        'Large_Leak': "🚨 Significant leak detected - urgent repair needed",
        'Corrosion': "🔧 Pipe corrosion detected - replace pipe section"
    }
    return descriptions.get(anomaly, "Unknown anomaly type")

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 Pipeline Anomaly Detection System</h1>', unsafe_allow_html=True)
    st.markdown(f"**User:** nischal2805 | **Date:** 2025-05-25 16:22:14 UTC")
    
    # Load model
    model, scaler, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Model not found! Please run `python anomaly_prediction_model.py` first to train the model.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for input parameters
    st.sidebar.header("🎛 Pipeline Parameters")
    
    # Input mode selection
    input_mode = st.sidebar.radio(
        "📊 Input Mode",
        ["Calculate Pressure", "Direct Pressure Input"],
        help="Choose whether to calculate pressure from physics or input it directly"
    )
    
    # Input controls
    flow_rate = st.sidebar.slider(
        "Flow Rate (m³/s)", 
        min_value=0.01, max_value=3.0, value=1.0, step=0.01,
        help="Water flow rate through the pipeline"
    )
    
    diameter = st.sidebar.slider(
        "Pipe Diameter (m)", 
        min_value=0.05, max_value=1.0, value=0.3, step=0.01,
        help="Internal diameter of the pipe"
    )
    
    length = st.sidebar.slider(
        "Pipe Length (m)", 
        min_value=100, max_value=5000, value=1000, step=50,
        help="Total length of the pipeline"
    )
    
    roughness = st.sidebar.slider(
        "Surface Roughness (m)", 
        min_value=0.00001, max_value=0.001, value=0.0001, step=0.00001, format="%.5f",
        help="Surface roughness of pipe material"
    )
    
    # Conditional pressure input
    input_pressure = None
    if input_mode == "Direct Pressure Input":
        input_pressure = st.sidebar.slider(
            "Pressure Drop (Pa)", 
            min_value=1000, max_value=10000000, value=500000, step=1000,
            help="Direct pressure drop measurement in Pascals"
        )
        
        # Convert to kPa for display
        st.sidebar.caption(f"**{input_pressure/1000:.1f} kPa**")
    
    # Preset scenarios
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Quick Test Scenarios")
    
    col_preset1, col_preset2 = st.sidebar.columns(2)
    
    with col_preset1:
        if st.button("🟢 Normal", use_container_width=True):
            flow_rate = 1.2
            diameter = 0.3
            length = 1500
            roughness = 0.00008
            input_pressure = 400000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
        
        if st.button("🟡 Partial Block", use_container_width=True):
            flow_rate = 0.8
            diameter = 0.2
            length = 1500
            roughness = 0.00008
            input_pressure = 800000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
        
        if st.button("🟠 Small Leak", use_container_width=True):
            flow_rate = 0.6
            diameter = 0.3
            length = 1500
            roughness = 0.00008
            input_pressure = 200000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
    
    with col_preset2:
        if st.button("🔴 Major Block", use_container_width=True):
            flow_rate = 0.5
            diameter = 0.15
            length = 1500
            roughness = 0.00008
            input_pressure = 1200000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
        
        if st.button("🔴 Large Leak", use_container_width=True):
            flow_rate = 0.4
            diameter = 0.3
            length = 1500
            roughness = 0.00008
            input_pressure = 150000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
        
        if st.button("🟣 Corrosion", use_container_width=True):
            flow_rate = 1.0
            diameter = 0.3
            length = 1500
            roughness = 0.0005
            input_pressure = 600000 if input_mode == "Direct Pressure Input" else None
            st.rerun()
    
    # Main prediction section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Real-time Anomaly Detection")
        
        # Calculate features and predict
        features = calculate_features(flow_rate, diameter, length, roughness, input_pressure)
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities.max()
        
        # Display prediction
        color = get_anomaly_color(prediction)
        description = get_anomaly_description(prediction)
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
            <h2 style="color: {color}; margin: 0;">🎯 Prediction: {prediction}</h2>
            <h3 style="color: {color}; margin: 5px 0;">Confidence: {confidence:.1%}</h3>
            <p style="margin: 10px 0 0 0; font-size: 1.1em;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input method indicator
        pressure_source = "📊 Direct Input" if input_pressure is not None else "🧮 Calculated"
        st.caption(f"**Pressure Source:** {pressure_source}")
        
        # Probability chart
        st.subheader("📊 Prediction Probabilities")
        classes = model.classes_
        prob_df = pd.DataFrame({
            'Anomaly Type': classes,
            'Probability': probabilities
        }).sort_values('Probability', ascending=True)
        
        fig = px.bar(prob_df, x='Probability', y='Anomaly Type', 
                     orientation='h', color='Probability',
                     color_continuous_scale='RdYlGn_r',
                     title="Probability Distribution for Each Anomaly Type")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 predictions
        st.subheader("🏆 Top 3 Predictions")
        top_3 = prob_df.tail(3)
        for idx, row in top_3.iterrows():
            prob_percent = row['Probability'] * 100
            anomaly_type = row['Anomaly Type']
            color = get_anomaly_color(anomaly_type)
            
            st.markdown(f"""
            <div class="feature-box">
                <strong style="color: {color};">{anomaly_type}</strong>: {prob_percent:.1f}%
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚙ Calculated Parameters")
        
        # Display calculated values
        velocity = features[4]
        reynolds = features[5]
        pressure_drop = features[6]
        friction_factor = features[10]
        
        st.metric("🌊 Velocity", f"{velocity:.2f} m/s")
        st.metric("🔢 Reynolds Number", f"{reynolds:.0f}")
        
        # Highlight pressure with source indicator
        pressure_kpa = pressure_drop/1000
        pressure_label = "📉 Pressure Drop"
        if input_pressure is not None:
            pressure_label += " (Input)"
        else:
            pressure_label += " (Calculated)"
        st.metric(pressure_label, f"{pressure_kpa:.1f} kPa")
        
        st.metric("⚙ Friction Factor", f"{friction_factor:.4f}")
        
        # Flow regime
        if reynolds < 2300:
            flow_regime = "🐌 Laminar"
            regime_color = "#28a745"
        elif reynolds < 4000:
            flow_regime = "🌪 Transition"
            regime_color = "#ffc107"
        else:
            flow_regime = "🌊 Turbulent"
            regime_color = "#17a2b8"
        
        st.markdown(f"""
        <div style="background-color: {regime_color}20; padding: 10px; border-radius: 8px; border-left: 4px solid {regime_color};">
            <strong>Flow Regime:</strong> <span style="color: {regime_color};">{flow_regime}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Analysis Section
    st.markdown("---")
    st.subheader("📈 Feature Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Pressure gradient visualization
        pressure_gradient = features[7]
        fig_pressure = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pressure_gradient,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Pressure Gradient (Pa/m)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 200], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 150}}))
        fig_pressure.update_layout(height=300)
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col4:
        # Efficiency indicator
        flow_efficiency = features[8]
        fig_efficiency = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = flow_efficiency * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Flow Efficiency (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 70], 'color': "red"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85}}))
        fig_efficiency.update_layout(height=300)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Batch Testing Section
    st.markdown("---")
    st.subheader("🧪 Batch Testing")
    
    if st.button("🎲 Run Random Test Batch"):
        st.write("Running 10 random pipeline configurations...")
        
        test_results = []
        np.random.seed(42)
        
        for i in range(10):
            # Generate random parameters
            rand_flow = np.random.uniform(0.2, 2.0)
            rand_diameter = np.random.uniform(0.1, 0.5)
            rand_length = np.random.uniform(500, 3000)
            rand_roughness = np.random.uniform(0.00005, 0.0003)
            
            # Predict
            rand_features = calculate_features(rand_flow, rand_diameter, rand_length, rand_roughness)
            rand_features_scaled = scaler.transform([rand_features])
            rand_prediction = model.predict(rand_features_scaled)[0]
            rand_confidence = model.predict_proba(rand_features_scaled)[0].max()
            
            test_results.append({
                'Test': i+1,
                'Flow Rate': round(rand_flow, 2),
                'Diameter': round(rand_diameter, 2),
                'Length': int(rand_length),
                'Roughness': round(rand_roughness, 5),
                'Prediction': rand_prediction,
                'Confidence': round(rand_confidence, 3)
            })
        
        # Display results
        results_df = pd.DataFrame(test_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary
        prediction_counts = results_df['Prediction'].value_counts()
        st.write("**Test Summary:**")
        for anomaly, count in prediction_counts.items():
            st.write(f"- {anomaly}: {count} cases ({count/10*100:.0f}%)")

if __name__ == "__main__":
    main()
