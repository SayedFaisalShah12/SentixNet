import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from model import SentixNet
from physics import newtonian_gravity

def load_trained_model(model_path, input_size=4, hidden_size=64):
    model = SentixNet(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_force_vs_distance(model, scaler_x, scaler_y):
    """
    Plots Force vs Distance for a fixed mass pair.
    """
    m1, m2 = 50.0, 50.0
    distances = np.linspace(1, 50, 100)
    
    # Normal Gravity Data
    X_normal = np.zeros((100, 4))
    X_normal[:, 0] = m1
    X_normal[:, 1] = m2
    X_normal[:, 2] = distances
    X_normal[:, 3] = 0 # Normal
    
    # Anti-Gravity Data
    X_anti = np.zeros((100, 4))
    X_anti[:, 0] = m1
    X_anti[:, 1] = m2
    X_anti[:, 2] = distances
    X_anti[:, 3] = 1 # Anti
    
    # Predict
    with torch.no_grad():
        # Scale and convert to tensor
        X_normal_scaled = torch.tensor(scaler_x.transform(X_normal), dtype=torch.float32)
        X_anti_scaled = torch.tensor(scaler_x.transform(X_anti), dtype=torch.float32)
        
        y_pred_normal_scaled = model(X_normal_scaled).numpy()
        y_pred_anti_scaled = model(X_anti_scaled).numpy()
        
        # Inverse Scale
        y_pred_normal = scaler_y.inverse_transform(y_pred_normal_scaled)
        y_pred_anti = scaler_y.inverse_transform(y_pred_anti_scaled)
        
    # Analytical True Values
    y_true_normal = [newtonian_gravity(m1, m2, r, anti_gravity=False) for r in distances]
    y_true_anti = [newtonian_gravity(m1, m2, r, anti_gravity=True) for r in distances]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(distances, y_true_normal, 'r--', label='Analytical Normal')
    plt.plot(distances, y_pred_normal, 'r-', label='SentixNet Normal')
    plt.plot(distances, y_true_anti, 'b--', label='Analytical Anti')
    plt.plot(distances, y_pred_anti, 'b-', label='SentixNet Anti')
    plt.title('Force vs Distance (Analytical vs Predicted)')
    plt.xlabel('Distance (r)')
    plt.ylabel('Force (F)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Zoomed in to see the anti-gravity better
    plt.plot(distances, y_pred_normal, 'r-', label='SentixNet Normal')
    plt.plot(distances, y_pred_anti, 'b-', label='SentixNet Anti')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('SentixNet Field Behavior (Zoomed)')
    plt.xlabel('Distance (r)')
    plt.ylabel('Force (F)')
    plt.ylim(-100, 100)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "models", "field_visualization.png")
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    # We need the scalers from train.py logic or save them.
    # For this demo, I'll quickly re-fit them in this script or modify train.py to save them.
    # Since I don't want to overcomplicate, I'll just re-run the scaling logic here.
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gravity_data.csv")
    df = pd.read_csv(data_path)
    X = df[['m1', 'm2', 'r', 'is_anti_gravity']].values
    y = df['force'].values.reshape(-1, 1)
    
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "sentixnet_model.pth")
    model = load_trained_model(model_path)
    
    plot_force_vs_distance(model, scaler_x, scaler_y)
