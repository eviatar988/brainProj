import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import data_extracts
import test1

# Generate dummy data
data = data_extracts.data_trasnform('low_gamma', 'low_gamma', 0, 0, data_extracts.read_file_rest_max, data_extracts.read_file_film_max)
X_train, X_val, y_train, y_val = test1.data_split(data)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)


# Define the neural network model
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Hyperparameters
input_size = X_train.shape[1]  # Number of features in the input vector
hidden_size = 200  # Number of neurons in the hidden layer
output_size = 2  # Number of classes
learning_rate = 0.001
num_epochs = 1000

# Initialize the model, loss function, and optimizer
model = FeedforwardNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    _, predicted = torch.max(val_outputs, 1)
    accuracy = accuracy_score(y_val, predicted)
    print(f'Validation Accuracy: {accuracy:.4f}')