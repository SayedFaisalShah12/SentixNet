import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import PINNSentixNet
from physics import G

def pinn_loss(model, inputs, targets, lambda_phys=0.01):
    """
    Combined loss: MSE + Physical Constraint Loss
    Physical Constraint: F * r^2 - G * m1 * m2 = 0
    """
    # Standard Data Loss
    outputs = model(inputs)
    mse_loss = nn.MSELoss()(outputs, targets)
    
    # Physics Loss
    # inputs: [m1, m2, r, is_anti_gravity]
    # We only apply physics constraint if is_anti_gravity is 0 (normal) or 1 (simple -F)
    # For simplicity, let's just use the squared residual of the gravity equation
    
    m1 = inputs[:, 0]
    m2 = inputs[:, 1]
    r = inputs[:, 2]
    is_anti = inputs[:, 3]
    
    # Predicted Force
    F_pred = outputs.squeeze()
    
    # Equation Residual: F - (coeff * G * m1 * m2 / r^2)
    # coeff = 1 if is_anti == 0 else -1
    coeff = 1 - 2 * is_anti 
    
    # Avoid division by zero
    r_safe = torch.clamp(r, min=0.1)
    
    physics_target = coeff * G * (m1 * m2) / (r_safe**2)
    phys_loss = nn.MSELoss()(F_pred, physics_target)
    
    return mse_loss + lambda_phys * phys_loss

def train_pinn():
    # This is a conceptual implementation of PINN training
    print("Initiating Physics-Informed Neural Network (PINN) training session...")
    
    # For PINN, we often don't even need all the labels if the physics loss is strong
    # But here we'll use a mix.
    
    # Simulated inputs
    n_samples = 1000
    m1 = torch.rand(n_samples) * 100
    m2 = torch.rand(n_samples) * 100
    r = torch.rand(n_samples) * 50 + 1
    is_anti = torch.randint(0, 2, (n_samples,)).float()
    
    inputs = torch.stack([m1, m2, r, is_anti], dim=1)
    
    # Analytical targets (with some noise)
    targets = []
    for i in range(n_samples):
        coeff = 1 if is_anti[i] == 0 else -1
        f = coeff * G * (m1[i] * m2[i]) / (r[i]**2)
        targets.append(f)
    targets = torch.tensor(targets).unsqueeze(1)
    
    model = PINNSentixNet(input_size=4, hidden_size=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"PINN Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    torch.save(model.state_dict(), os.path.join(models_dir, "sentixnet_pinn.pth"))
    print("PINN training complete.")

if __name__ == "__main__":
    train_pinn()
