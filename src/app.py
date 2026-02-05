import streamlit as st
import torch
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from model import SentixNet
from physics import newtonian_gravity, G

# Set page config
st.set_page_config(page_title="SentixNet: Anti-Gravity Simulator", layout="wide", page_icon="🚀")

# Load assets
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sentixnet_model.pth")
SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_y.joblib")

@st.cache_resource
def load_model():
    model = SentixNet(input_size=4, hidden_size=64)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

@st.cache_resource
def load_scalers():
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return scaler_X, scaler_y

# Title and Description
st.title("🌌 SentixNet: Anti-Gravity Field Simulation")
st.markdown("""
Welcome to **SentixNet**, an AI-powered physics simulator. This application uses a deep learning model to predict gravitational forces and explore the hypothetical behavior of anti-gravity fields.
""")

# Sidebar for Inputs
st.sidebar.header("🕹️ Simulation Controls")
m1 = st.sidebar.slider("Mass 1 (kg)", 1.0, 100.0, 50.0)
m2 = st.sidebar.slider("Mass 2 (kg)", 1.0, 100.0, 50.0)
r = st.sidebar.slider("Distance (m)", 1.0, 50.0, 10.0)
anti_gravity = st.sidebar.checkbox("Enable Anti-Gravity (Sentix Factor)")

# Model Prediction
try:
    model = load_model()
    scaler_X, scaler_y = load_scalers()

    # Prepare Input
    is_anti = 1 if anti_gravity else 0
    input_data = np.array([[m1, m2, r, is_anti]])
    input_scaled = torch.tensor(scaler_X.transform(input_data), dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(input_scaled).numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]

    # Analytical Calculation
    analytical = newtonian_gravity(m1, m2, r, anti_gravity=anti_gravity)

    # Layout: Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Force (N)", f"{prediction:.4f}")
    col2.metric("Analytical Force (N)", f"{analytical:.4f}")
    error = abs(prediction - analytical) / (abs(analytical) + 1e-9) * 100
    col3.metric("Prediction Error (%)", f"{error:.2f}%")

    # Plotting
    st.divider()
    st.subheader("📊 Dynamic Force Visualization")
    
    # Generate Curve Data
    distances = np.linspace(1, 50, 100)
    X_curve = np.zeros((100, 4))
    X_curve[:, 0] = m1
    X_curve[:, 1] = m2
    X_curve[:, 2] = distances
    X_curve[:, 3] = is_anti
    
    X_curve_scaled = torch.tensor(scaler_X.transform(X_curve), dtype=torch.float32)
    with torch.no_grad():
        y_curve_scaled = model(X_curve_scaled).numpy()
        y_curve_pred = scaler_y.inverse_transform(y_curve_scaled).flatten()
    
    y_curve_true = np.array([newtonian_gravity(m1, m2, d, anti_gravity=anti_gravity) for d in distances])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances, y=y_curve_true, name="Analytical Theory", line=dict(dash='dash', color='gray')))
    fig.add_trace(go.Scatter(x=distances, y=y_curve_pred, name="SentixNet Prediction", line=dict(color='cyan', width=3)))
    fig.add_trace(go.Scatter(x=[r], y=[prediction], name="Current State", marker=dict(size=12, color='red')))
    
    fig.update_layout(
        xaxis_title="Distance (m)",
        yaxis_title="Force (N)",
        template="plotly_dark",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.info("Please ensure you have run 'python src/train.py' first to generate model weights and scalers.")

# Step-by-Step Explanation Section
st.divider()
st.subheader("📖 How SentixNet Works (Step-by-Step)")

with st.expander("Step 1: Physics Formulation"):
    st.write("""
    We start with Newtonian Physics. The force is proportional to the product of masses and inversely proportional to the square of distance.
    In the **Standard** mode, the force is attractive. In **Anti-Gravity** mode, we flip the polarity of the field.
    """)
    st.latex(r"F = \pm G \frac{m_1 m_2}{r^2}")

with st.expander("Step 2: Synthetic Data Generation"):
    st.write("""
    We generated a dataset of 5,000 random interactions. 
    Each sample contains ($m_1, m_2, r, \text{field\_type}$) as features and the calculated Force as the target label. 
    We add Gaussian noise to simulate real-world sensor inaccuracies.
    """)

with st.expander("Step 3: Neural Network Training"):
    st.write("""
    We use a Deep Multi-Layer Perceptron (MLP) built with PyTorch. 
    The model consists of 3 hidden layers that learn to map the raw mass and distance values to the force. 
    The network 'discovered' the inverse-square law through backpropagation.
    """)

with st.expander("Step 4: Inference & Real-time Prediction"):
    st.write("""
    When you move the sliders above, the input features are standardized (scaled) and passed through the saved model weights. 
    The resulting output is then inverse-scaled back to physical units (Newtons) for display.
    """)

st.sidebar.markdown("---")
st.sidebar.info("Developed by SentixNet AI Research Team.")
