import torch
import torch.nn as nn

# NOTE: Deeper networks should be trained with smaller learning rates
# Even in this case of just adding one more hidden layer with increased feature size, the training fails when lr is set to the same lr used from before(which is lr=0.003)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, learning_rate=0.0003, num_iterations=3000):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.relu = nn.ReLU(inplace=True)
        
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size_1)
        self.hidden_layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.output_layer = nn.Linear(hidden_size_2, output_size)
        
    def forward(self, x):
        x = self.relu(self.hidden_layer_1(x))
        x = self.relu(self.hidden_layer_2(x))
        x = self.relu(self.output_layer(x))
        return x
    
    def fit(self, X, y):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        
        for _ in range(self.num_iterations):
            output = self.forward(X)
            loss = criterion(output.squeeze(), y) #Be careful of the right shape of y_hat and y.
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X)
        
# Make some sample data
X = torch.tensor([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=torch.float32)
y = torch.tensor([2*x_1 + 3*x_2 for x_1, x_2 in X], dtype=torch.float32)

# Create model
num_features = X.shape[1]
model = MLP(input_size=num_features, hidden_size_1=num_features*2, hidden_size_2 = num_features*4, output_size=1)

# Train model
model.fit(X, y)

# Run inference
X_new = torch.tensor([[6, 12], [7, 14]], dtype=torch.float32)
y_pred = model.predict(X_new)
Y_actual = torch.tensor([2*x_1 + 3*x_2 for x_1, x_2 in X_new], dtype=torch.float32)

print("== Results ==")
print(f"Answers: {Y_actual.numpy()}")
print(f"Predictions: {y_pred.squeeze().numpy()}")
print(f"Avg. Error: {torch.mean(torch.abs(y_pred.squeeze() - Y_actual)).item()}")
print()