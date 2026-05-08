import numpy as np
import matplotlib.pyplot as plt

# 1. Setup our parameters
T = 20  # Sequence length (e.g., 20 words or 20 days)
weights = [0.5, 0.8, 1.0, 1.1, 1.5] # From vanishing to exploding

# Time steps going backward from T (20) down to 0
time_steps = np.arange(T, -1, -1) 

plt.figure(figsize=(10, 6))

# 2. Simulate the gradient flow for each weight
for w in weights:
    # The gradient at the final step T is 1.0. 
    # At any previous step t, it is 1.0 * w^(T - t)
    gradients = [1.0 * (w ** (T - t)) for t in time_steps]
    
    # Plot the line for this specific weight
    plt.plot(time_steps, gradients, marker='o', label=f'W = {w}')

# 3. Format the chart
plt.yscale('log')  # CRITICAL: Use log scale to see both tiny and huge numbers
plt.title('BPTT: Vanishing and Exploding Gradients (Log Scale)')
plt.xlabel('Time Step (Moving Backward from T=20 to 0)')
plt.ylabel('Gradient Magnitude (Log Scale)')

# Invert the X-axis so it reads visually as "flowing backward"
plt.gca().invert_xaxis() 

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()