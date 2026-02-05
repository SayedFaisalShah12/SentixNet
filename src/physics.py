import numpy as np

# Universal Gravitational Constant (in normalized simulation units)
G = 1.0 

def newtonian_gravity(m1, m2, r, anti_gravity=False):
    """
    Computes the force of gravity between two masses.
    
    Args:
        m1 (float/np.array): Mass of object 1
        m2 (float/np.array): Mass of object 2
        r (float/np.array): Distance between centers of mass
        anti_gravity (bool): If True, returns a repulsive force (anti-gravity)
        
    Returns:
        float/np.array: The gravitational force
    """
    # Ensure r is not zero to avoid division by zero
    r = np.maximum(r, 0.1) 
    
    force = G * (m1 * m2) / (r**2)
    
    if anti_gravity:
        return -force # Repulsive force
    return force # Attractive force

def get_acceleration(force, mass):
    """F = ma => a = F/m"""
    return force / mass
