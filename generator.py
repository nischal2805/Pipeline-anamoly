import pandas as pd
import numpy as np
import math

def calculate_friction_factor(reynolds, relative_roughness):
    """More accurate friction factor calculation using Colebrook-White equation"""
    if reynolds < 2300:
        return 64 / reynolds  # Laminar flow
    elif reynolds < 4000:  # Transition zone
        return 0.032  # Approximation for transition
    else:
        # Iterative solution for Colebrook-White equation
        f_guess = 0.02
        for _ in range(10):  # Iterative solution
            f_new = 1 / (-2 * math.log10(relative_roughness/3.7 + 2.51/(reynolds * math.sqrt(f_guess))))**2
            if abs(f_new - f_guess) < 1e-6:
                break
            f_guess = f_new
        return f_new

def add_sensor_noise(value, noise_percent=2.0):
    """Add realistic sensor measurement noise"""
    noise = np.random.normal(0, noise_percent/100)
    return value * (1 + noise)

def add_temporal_effects(base_value, anomaly_type, severity_factor):
    """Simulate how anomalies develop over time"""
    if anomaly_type == 'Corrosion':
        # Gradual surface roughness increase
        degradation = 1 + (severity_factor * 0.15)
        return base_value * degradation
    elif 'Blockage' in anomaly_type:
        # Progressive diameter reduction
        blockage_factor = severity_factor * 0.3
        return base_value * (1 - blockage_factor)
    elif 'Leak' in anomaly_type:
        # Flow rate reduction due to leak
        leak_factor = severity_factor * 0.2
        return base_value * (1 - leak_factor)
    return base_value

def generate_realistic_pipeline_dataset(n_samples=3000):
    """Generate realistic pipeline dataset with proper physics and noise"""
    data = []
    
    for _ in range(n_samples):
        # Base pipeline parameters (realistic ranges)
        flow_rate = np.random.uniform(0.1, 2.0)      # m³/s
        diameter = np.random.uniform(0.15, 0.5)      # m  
        length = np.random.uniform(500, 3000)        # m
        roughness = np.random.uniform(0.000045, 0.00015)  # m (steel pipe)
        fluid_density = 1000  # kg/m³ (water)
        fluid_viscosity = 0.001  # Pa·s (water at 20°C)
        
        # Choose anomaly type with realistic probabilities
        anomaly = np.random.choice([
            'Normal', 'Partial_Blockage', 'Major_Blockage', 
            'Small_Leak', 'Large_Leak', 'Corrosion'
        ], p=[0.5, 0.15, 0.1, 0.1, 0.1, 0.05])
        
        # Severity factor for temporal effects
        severity_factor = np.random.uniform(0.1, 1.0)
        
        # Apply realistic anomaly effects
        effective_diameter = diameter
        effective_flow = flow_rate
        effective_roughness = roughness
        
        if anomaly == 'Partial_Blockage':
            effective_diameter = add_temporal_effects(diameter, anomaly, severity_factor * 0.5)
        elif anomaly == 'Major_Blockage':
            effective_diameter = add_temporal_effects(diameter, anomaly, severity_factor)
        elif anomaly == 'Small_Leak':
            effective_flow = add_temporal_effects(flow_rate, anomaly, severity_factor * 0.5)
        elif anomaly == 'Large_Leak':
            effective_flow = add_temporal_effects(flow_rate, anomaly, severity_factor)
        elif anomaly == 'Corrosion':
            effective_roughness = add_temporal_effects(roughness, anomaly, severity_factor)
        
        # Calculate flow parameters
        cross_sectional_area = math.pi * (effective_diameter/2)**2
        velocity = effective_flow / cross_sectional_area
        reynolds = fluid_density * velocity * effective_diameter / fluid_viscosity
        relative_roughness = effective_roughness / effective_diameter
        
        # Calculate friction factor using proper physics
        friction_factor = calculate_friction_factor(reynolds, relative_roughness)
        
        # Darcy-Weisbach equation for head loss
        head_loss = friction_factor * (length/effective_diameter) * (velocity**2/(2*9.81))
        pressure_drop = fluid_density * 9.81 * head_loss  # Pascals
        
        # Additional engineered features
        pressure_gradient = pressure_drop / length
        flow_efficiency = effective_flow / flow_rate if flow_rate > 0 else 1.0
        reynolds_number_normalized = reynolds / 100000  # Normalize for ML
        velocity_head = (velocity**2) / (2 * 9.81)
        
        # Add realistic sensor noise (2-3% typical for industrial sensors)
        flow_rate_measured = add_sensor_noise(effective_flow, 1.5)
        pressure_drop_measured = add_sensor_noise(pressure_drop, 2.0)
        velocity_measured = add_sensor_noise(velocity, 1.0)
        
        # Anomaly severity indicator (0-1 scale)
        anomaly_severity = severity_factor if anomaly != 'Normal' else 0.0
        
        data.append([
            flow_rate_measured, diameter, length, effective_roughness, 
            velocity_measured, reynolds, pressure_drop_measured, 
            pressure_gradient, flow_efficiency, reynolds_number_normalized,
            friction_factor, velocity_head, anomaly_severity, anomaly
        ])
    
    columns = [
        'flow_rate', 'diameter', 'length', 'roughness', 'velocity', 
        'reynolds', 'pressure_drop', 'pressure_gradient', 'flow_efficiency',
        'reynolds_normalized', 'friction_factor', 'velocity_head', 
        'anomaly_severity', 'anomaly'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate the enhanced dataset
print("Generating realistic pipeline anomaly detection dataset...")
df = generate_realistic_pipeline_dataset(3000)

# Save dataset
df.to_csv('pipeline_anomaly_data.csv', index=False)

print(f"✓ Dataset created with {len(df)} samples")
print(f"✓ Dataset saved as 'pipeline_anomaly_data.csv'")
print("\nAnomaly Distribution:")
print(df['anomaly'].value_counts())
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 samples:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())