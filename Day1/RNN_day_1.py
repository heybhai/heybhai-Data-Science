import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Internal weights
        self.W_xh = nn.Linear(input_size, hidden_size) # Input to hidden
        self.W_hh = nn.Linear(hidden_size, hidden_size) # Hidden to hidden
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        """
        x: current input (batch, input_size)
        h_prev: previous hidden state (batch, hidden_size)
        """
        # The core RNN math:
        h_t = self.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h_t

# --- Simulation ---
input_dim = 10   # e.g., vector size of a word
hidden_dim = 20  # internal memory size
seq_len = 5      # number of words in a sentence

model = SimpleRNN(input_dim, hidden_dim)
h = torch.zeros(1, hidden_dim) # Initial memory is zeros
sequence = torch.randn(seq_len, 1, input_dim) # Random sequence of 5 words

# Processing the sequence step-by-step
for i in range(seq_len):
    x_t = sequence[i]
    h = model(x_t, h)
    print(f"Step {i+1} hidden state (first 5 values): {h[0][:5].detach().numpy()}")