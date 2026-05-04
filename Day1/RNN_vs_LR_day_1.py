import torch
import torch.nn as nn

# Data: 9 days of prices for AlphaCorp
data_raw = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# --- 1. Linear Regression Output ---
# Input is a flat row of 9 numbers
lin_reg = nn.Linear(9, 1)
output_reg = lin_reg(data_raw)

# --- 2. RNN Output ---
# Input is a sequence of 9 steps
rnn_cell = nn.RNN(input_size=1, hidden_size=4, batch_first=True)
fc_layer = nn.Linear(4, 1)

# Reshape data for RNN: [Batch, Seq_Len, Input_Dim] -> [1, 9, 1]
rnn_input = data_raw.view(1, 9, 1)
rnn_hidden_out, _ = rnn_cell(rnn_input)
output_rnn = fc_layer(rnn_hidden_out[:, -1, :]) # Take only the last day's memory

print(f"Linear Regression Prediction: {output_reg.item():.4f}")
print(f"RNN Prediction:                {output_rnn.item():.4f}")
