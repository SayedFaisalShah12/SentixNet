# SentixNet: Anti-Gravity Field Simulation using Deep Learning

SentixNet is a physics-inspired deep learning project designed to simulate and predict gravitational force behaviors, including hypothetical anti-gravity fields. By combining classical Newtonian mechanics with modern neural network architectures, SentixNet demonstrates how AI can learn complex physical laws from synthetic data.

## 🚀 Project Overview
The goal of SentixNet is to bridge the gap between traditional physics simulations and machine learning. We use a **Fully Connected Neural Network (MLP)** and explore **Physics-Informed Neural Networks (PINNs)** to model the relationship between mass, distance, and force.

### Key Features:
- **Synthetic Data Generation**: Realistic gravity datasets based on $F = G \frac{m_1 m_2}{r^2}$.
- **Anti-Gravity Hypothesis**: Modeling repulsive forces where $F < 0$ for specialized field types.
- **Deep Learning Model**: PyTorch-based regression model to predict Force $F$.
- **PINN Integration**: Leveraging physical laws in the loss function to improve generalization.
- **Advanced Visualizations**: Comparative analysis of analytical vs. predicted results.

---

## 🔬 Mathematical Formulation

### 1. Newtonian Gravity
The attractive force between two masses $m_1$ and $m_2$ separated by distance $r$ is:
$$F = G \frac{m_1 m_2}{r^2}$$
*Where $G$ is the gravitational constant.*

### 2. Sentix Anti-Gravity Assumption
In this simulation, we introduce a boolean parameter $S \in \{0, 1\}$. When $S=1$, the force becomes repulsive:
$$F = -G \frac{m_1 m_2}{r^2}$$
This serves as the basis for training our model to distinguish between different field behaviors.

---

## 🛠️ Project Structure

```text
SentixNet/
├── data/               # Generated datasets (CSV)
├── models/             # Saved PyTorch models and plots
├── src/
│   ├── physics.py      # Core physical equations
│   ├── data_gen.py     # Synthetic data engine
│   ├── model.py        # Neural Network architectures
│   ├── train.py        # Standard MLP training loop
│   ├── pinn_train.py   # Physics-Informed training loop
│   └── visualize.py    # Analytical vs Prediction plots
└── README.md           # This documentation
```

---

## 📈 Model Architecture

SentixNet uses a deep MLP with the following architecture:
- **Input Layer**: 4 features (Mass 1, Mass 2, Distance, Field Type)
- **Hidden Layers**: 3 Fully Connected layers (64, 128, 64 neurons)
- **Activation**: ReLU (for data-driven) or Tanh (for PINNs)
- **Output Layer**: 1 neuron (Predicted Gravitational Force)

---

## 🧪 Results & Evaluation

The model successfully learns the inverse-square law behavior. Even without being explicitly told the formula (in the standard MLP case), it approximates the curve $1/r^2$ and correctly identifies the sign change for anti-gravity fields.

### Visualizations
- **Force vs Distance**: Shows the characteristic curve.
- **Field Behavior**: Visualizes the attraction/repulsion transition.

*(Reference plots can be found in the `models/` directory)*

---

## 🎓 Educational Insights

1. **Inverse Square Law**: The model learns that as $r$ increases, $F$ drops exponentially.
2. **Feature Scaling**: Standardizing inputs is critical because $F$ varies across many orders of magnitude.
3. **PINNs**: By adding the equation $F - \text{physics\_target} = 0$ to the loss, we ensure the model doesn't just "guess" but actually follows the laws of physics.

---

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate data: `python src/data_gen.py`
3. Train model: `python src/train.py`
4. Visualize: `python src/visualize.py`

---

## 🚧 Limitations & Future Work
- **Relativistic Effects**: Currently ignores Einstein’s General Relativity.
- **Dynamic Bodies**: Future iterations could simulate N-body problems.
- **Explainability**: Using SHAP or LIME to interpret how the model "understands" mass vs distance.

---
*Developed as a Research-Style Deep Learning Project by Antigravity AI.*
