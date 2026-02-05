import numpy as np
import pandas as pd
import os
from physics import newtonian_gravity

def generate_gravitational_data(n_samples=5000, seed=42):
    """
    Generates a synthetic dataset for gravitational and anti-gravitational forces.
    """
    np.random.seed(seed)
    
    # Randomly sample masses (1 to 100 units)
    m1 = np.random.uniform(1, 100, n_samples)
    m2 = np.random.uniform(1, 100, n_samples)
    
    # Randomly sample distances (1 to 50 units)
    r = np.random.uniform(1, 50, n_samples)
    
    # 50% chance of being anti-gravity
    is_anti_gravity = np.random.choice([0, 1], size=n_samples)
    
    forces = []
    for i in range(n_samples):
        f = newtonian_gravity(m1[i], m2[i], r[i], anti_gravity=bool(is_anti_gravity[i]))
        forces.append(f)
    
    # Convert to numpy array
    forces = np.array(forces)
    
    # Add some Gaussian noise to simulate 'measurement error'
    noise = np.random.normal(0, 0.05 * np.abs(forces), n_samples)
    forces_with_noise = forces + noise
    
    df = pd.DataFrame({
        'm1': m1,
        'm2': m2,
        'r': r,
        'is_anti_gravity': is_anti_gravity,
        'force': forces_with_noise,
        'true_force': forces # Keep for evaluation
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic gravitational data...")
    df = generate_gravitational_data()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    save_path = os.path.join(data_dir, "gravity_data.csv")
    df.to_csv(save_path, index=False)
    
    print(f"Dataset saved to {save_path}")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
