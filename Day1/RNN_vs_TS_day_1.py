# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Generate 30 days of data
# np.random.seed(42)
# days = 30
# start_price = 100
# # Daily returns: Mean 0.1%, Volatility 1.5%
# returns = np.random.normal(0.001, 0.015, days) 
# price_list = [start_price]

# for i in range(days):
#     price_list.append(price_list[-1] * (1 + returns[i]))

# df = pd.DataFrame({'Day': np.arange(days + 1), 'Price': price_list})
# plt.plot(df['Day'], df['Price'])
# plt.title("30-Day Synthetic Stock Price")
# plt.show()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# Generate 30 days of data
np.random.seed(42)
days = 200
prices = [100]
for _ in range(days):
    prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))

df = pd.DataFrame({'Price': prices}, index=pd.date_range(start='2024-01-01', periods=days + 1))
plt.plot(df.index, df['Price'])
plt.title("200-Day Synthetic Stock Price")
plt.show()
# Fit ARIMA (1,1,1)
# Note: we use all 31 points to predict the 32nd
arima_model = ARIMA(df['Price'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast the next day
arima_forecast = arima_result.get_forecast(steps=1)
print(f"ARIMA Next Day Prediction: {arima_forecast.predicted_mean.values[0]:.2f}")

# 1. Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Price']].values)

# Create sequences (e.g., use 5 days to predict the 6th)
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

window_size = 5
X, y = create_sequences(scaled_data, window_size)

# 2. Define RNN Architecture
class StockRNN(nn.Module):
    def __init__(self):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Output of the last time step

model = StockRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. Quick Training Loop (100 epochs)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# 4. Predict Next Day
model.eval()
last_window = torch.tensor(scaled_data[-window_size:], dtype=torch.float32).view(1, window_size, 1)
rnn_pred_scaled = model(last_window)
rnn_forecast = scaler.inverse_transform(rnn_pred_scaled.detach().numpy())
print(f"RNN Next Day Prediction:   {rnn_forecast[0][0]:.2f}")

# 5. Define the 'Next Day' timestamp
next_day = df.index[-1] + pd.Timedelta(days=1)

# 6. Visualization
plt.figure(figsize=(12, 6))

# Plot the historical 200-day data
plt.plot(df.index, df['Price'], label='Historical Data', color='black', linewidth=1.5)

# Plot ARIMA Prediction
plt.scatter(next_day, arima_forecast.predicted_mean.values[0], 
            color='red', marker='o', s=100, label=f'ARIMA Prediction: {arima_forecast.predicted_mean.values[0]:.2f}')

# Plot RNN Prediction
plt.scatter(next_day, rnn_forecast[0][0], 
            color='blue', marker='x', s=100, linewidths=3, label=f'RNN Prediction: {rnn_forecast[0][0]:.2f}')

# Formatting the chart
plt.title("ARIMA vs RNN: Next Day Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()