import numpy as np

# 1. Setup our minimalist RNN parameters
T = 10                  # Sequence length (10 days/words)
W_hh = 0.8              # The shared hidden weight
h = np.zeros(T + 1)     # Array to store hidden states
h[0] = 1.0              # Initial hidden state (Day 0)

# --- FORWARD PASS ---
print("--- Forward Pass (Hidden States) ---")
for t in range(1, T + 1):
    # Simplified RNN: new_memory = Weight * old_memory
    h[t] = W_hh * h[t-1]
    print(f"Time {t}: h = {h[t]:.4f}")

# --- BACKWARD PASS (BPTT) ---
print("\n--- Backward Pass (Gradient Flow) ---")
# Assume the "Loss" at the very end is 1.0. We want to see how much of 
# this loss makes it back to the beginning to update the weights.
gradient_at_T = 1.0 

# Array to store the gradient as it flows back
grad = np.zeros(T + 1)
grad[T] = gradient_at_T

# Flowing backward from T to 1
for t in reversed(range(1, T + 1)):
    # The gradient at the previous step is the current gradient * W_hh
    grad[t-1] = grad[t] * W_hh
    print(f"Gradient reaching Time {t-1}: {grad[t-1]:.6f}")

print("\nConclusion:")
print(f"The error signal reaching Day 0 is only {grad[0]*100:.2f}% of the original error.")


'''
Backpropagation Through Time (BPTT) & The Math of ForgettingTo understand BPTT, we have to look at the Chain Rule from calculus. 
Because an RNN uses the exact same weight matrix ($W_{hh}$) at every time step, an error at the end of the sentence (e.g., $t=10$) has to travel back through the network,
multiplying by that same weight matrix at every single step to update the weights for $t=1$.The Core EquationIf we simplify an RNN to remove the inputs and the activation function, the hidden state at time $T$ is just repeated multiplication:$$h_T = W_{hh} \cdot h_{T-1}$$To find out how much the first hidden state ($h_1$) influenced the final output ($h_T$),
we take the derivative using the chain rule:$$\frac{\partial h_T}{\partial h_1} = \frac{\partial h_T}{\partial h_{T-1}} \cdot \frac{\partial h_{T-1}}{\partial h_{T-2}} \dots \frac{\partial h_2}{\partial h_1}$$Because $\frac{\partial h_t}{\partial h_{t-1}} \approx W_{hh}$, this whole equation simplifies to:$$\frac{\partial h_T}{\partial h_1} \approx (W_{hh})^{T-1}$$
The Vanishing Gradient: If $W_{hh} = 0.9$ and your sentence is 30 words long, the gradient reaching the first word is $0.9^{30} = 0.04$. The signal is completely dead.


'''