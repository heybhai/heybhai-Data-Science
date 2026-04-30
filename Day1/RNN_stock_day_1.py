import torch
import torch.nn as nn

# 1. Define the RNN Model
class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        # input_size=1 (price), hidden_size=4 (internal memory)
        self.rnn = nn.RNN(input_size=1, hidden_size=4, batch_first=True)
        # Linear layer to map hidden state to a single price prediction
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        # h0 is the initial 'hidden state' (all zeros) [00:06:18]
        out, h_n = self.rnn(x) 
        # We only care about the very last output (Day 10 prediction) [00:10:25]
        prediction = self.fc(out[:, -1, :])
        return prediction

# 2. Prepare dummy data (9 days of prices)
# Shape: (Batch_Size, Sequence_Length, Input_Dim)
alpha_corp = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]])
beta_tech = torch.tensor([[[0.5], [0.1], [0.8], [0.2], [0.9], [0.1], [0.7], [0.2], [0.8]]])

model = StockPredictor()

# 3. Get Predictions
with torch.no_grad():
    pred_alpha = model(alpha_corp)
    pred_beta = model(beta_tech)

print(f"AlphaCorp Day 10 Prediction: {pred_alpha.item():.2f}")
print(f"BetaTech Day 10 Prediction: {pred_beta.item():.2f}")