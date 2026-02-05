import torch
import torch.nn as nn
import torch.nn.functional as F

class SentixNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        """
        Fully connected neural network to predict gravitational force.
        
        Args:
            input_size (int): Number of input features (m1, m2, r, is_anti_gravity)
            hidden_size (int): Number of neurons in hidden layers
        """
        super(SentixNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # Output is Predicted Force
        )
        
    def forward(self, x):
        return self.network(x)

class PINNSentixNet(nn.Module):
    """
    Physics-Informed Neural Network (PINN) version.
    This architecture is similar but designed to handle physics constraints 
    during the training process (usually implemented in the loss function).
    """
    def __init__(self, input_size=4, hidden_size=64):
        super(PINNSentixNet, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(), # Tanh is often preferred for PINNs as it has smoother gradients
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.network(x)
