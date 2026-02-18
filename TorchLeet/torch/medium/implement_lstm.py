import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
y = torch.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1):
        super(CustomLSTMModel, self).__init__()
        
        self.hidden_units = hidden_units
        
        # Input weights
        self.W_f = nn.Linear(input_dim, hidden_units)
        self.W_i = nn.Linear(input_dim, hidden_units)
        self.W_o = nn.Linear(input_dim, hidden_units)
        self.W_c = nn.Linear(input_dim, hidden_units)

        # Hidden weights
        self.U_f = nn.Linear(hidden_units, hidden_units, bias=False)
        self.U_i = nn.Linear(hidden_units, hidden_units, bias=False)
        self.U_o = nn.Linear(hidden_units, hidden_units, bias=False)
        self.U_c = nn.Linear(hidden_units, hidden_units, bias=False)

        # Fully connected layer
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, inputs, H_C=None):
        batch_size, seq_len, _ = inputs.size()
        
        if H_C is None:
            h_t = torch.zeros(batch_size, self.hidden_units)
            c_t = torch.zeros(batch_size, self.hidden_units)
        else:
            h_t, c_t = H_C

        outputs = []

        for t in range(seq_len):
            x_t = inputs[:, t, :]

            f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_t))
            i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_t))
            o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_t))
            c_hat_t = torch.tanh(self.W_c(x_t) + self.U_c(h_t))

            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        predictions = self.fc(outputs)

        return predictions, (h_t, c_t)
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_units=50, output_dim=1):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize the model, loss function, and optimizer
model_custom = CustomLSTMModel(1, 50)
model_inbuilt = LSTMModel()
criterion = nn.MSELoss()
optimizer_custom = optim.Adam(model_custom.parameters(), lr=0.01)
optimizer_inbuilt = optim.Adam(model_inbuilt.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    # Forward pass
    state = None
    pred, state = model_custom(X_seq, state)
    loss = criterion(pred[:, -1, :], y_seq) # Use the last output of the LSTM
    # Backward pass and optimization
    optimizer_custom.zero_grad()
    loss.backward()
    optimizer_custom.step()

    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        

# Training loop for the inbuilt model
epochs = 500
for epoch in range(epochs):
    # Forward pass
    pred = model_inbuilt(X_seq)
    loss = criterion(pred, y_seq)
    # Backward pass and optimization
    optimizer_inbuilt.zero_grad()
    loss.backward()
    optimizer_inbuilt.step()

    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        
# Testing on new data
test_steps = 100  # Ensure this is greater than sequence_length
X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
y_test = torch.sin(X_test)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

with torch.no_grad():
    pred_custom, _ = model_custom(X_test_seq)
    pred_inbuilt = model_inbuilt(X_test_seq)
pred_custom = torch.flatten(pred_custom[:, -1, :])
pred_inbuilt = pred_inbuilt.squeeze()
print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")


#Plot the predictions
plt.figure()
# plt.plot(y_test, label="Ground Truth")
plt.plot(pred_custom, label="custom model")
plt.plot(pred_inbuilt, label="inbuilt model")
plt.legend()
plt.show()