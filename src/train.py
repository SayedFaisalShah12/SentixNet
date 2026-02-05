import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from model import SentixNet

def train_model():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "gravity_data.csv")
    df = pd.read_csv(data_path)
    
    # 2. Preprocess Data
    # Features: m1, m2, r, is_anti_gravity
    X = df[['m1', 'm2', 'r', 'is_anti_gravity']].values
    y = df['force'].values.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (Gravity data can have large ranges)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale Target (Force can also be large)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                  torch.tensor(y_train_scaled, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                 torch.tensor(y_test_scaled, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. Initialize Model, Loss, Optimizer
    model = SentixNet(input_size=4, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 50
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_batch_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # 5. Save Model and Scalers
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(models_dir, "sentixnet_model.pth"))
    
    import joblib
    joblib.dump(scaler_X, os.path.join(models_dir, "scaler_X.joblib"))
    joblib.dump(scaler_y, os.path.join(models_dir, "scaler_y.joblib"))
    
    print("Training complete. Model and scalers saved.")
    
    # 6. Plot Training History
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('SentixNet Training History')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(models_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    
    return model, scaler_X, scaler_y

if __name__ == "__main__":
    train_model()
