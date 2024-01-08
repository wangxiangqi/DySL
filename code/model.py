import torch
import torch.nn as nn
import torch.optim as optim

# Define your custom neural network class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Assuming you have data with several features (y) and corresponding labels (x)
# X_train and Y_train represent your input features and output labels, respectively.

# Convert your data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
Y_train_tensor = torch.FloatTensor(Y_train)

# Create an instance of your MLP model
input_size = X_train.shape[1]  # Number of features in your input (y)
hidden_size1 = 64
hidden_size2 = 32
output_size = number_of_output_features  # Number of output features (x)

model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)

    # Compute the loss
    loss = criterion(outputs, Y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Now your model is trained, and you can use it for predictions
