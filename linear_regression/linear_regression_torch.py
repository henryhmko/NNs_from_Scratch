import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, num_features, learning_rate=0.003, num_iterations=1000):
        super(LinearRegression, self).__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # in_features is num_features
        # out_features is 1 since we want to get a class
        self.linear = nn.Linear(num_features, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x) # Pass x into nn.Linear that was initialized in __init__()
    
    def fit(self, X, y):
        num_samples, num_features = X.shape # X is a mxn matrix where m is num_samples and n is num_features
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        for _ in range(self.num_iterations):
            # 1. Predict y_hat
            # 2. Compute loss
            # 3. Update params based on loss gradients
                # a. zero grad optimizer
                # b. compute loss gradients
                # c. update optimizer based on gradients
            # 1. predict y_hat
            y_hat = self.forward(X)
            # 2. Compute loss
            loss = criterion(y_hat.squeeze(), y) # y_hat.shape = [batch_size, 1], but y.shape = [batch_size,]
            # 3. Update params based on loss gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X)

# Run tests
# 1. Make some sample data with only one feature.
# i.e. x_i.shape = (1,)
X = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([x*10 for x in range(1, 6)], dtype=torch.float32)

# Create model
model = LinearRegression(num_features=1)

# Train
model.fit(X, y)

# Run Inference
X_new = torch.tensor([[6], [7]], dtype=torch.float32)
y_pred = model.predict(X_new)
Y_actual = torch.tensor([60, 70], dtype=torch.float32)

print("== Results for 1 Feature ==")
print(f"Predictions: {y_pred.squeeze()}")
print(f"Avg. Error: {torch.mean(torch.abs(y_pred.squeeze() - Y_actual)).item()}")
print()

# 2. Make some sample data with two features.
X = torch.tensor([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=torch.float32)
y = torch.tensor([2*x_1 + 3*x_2 for x_1, x_2 in X], dtype=torch.float32)

# Create model
model = LinearRegression(num_features=2)

# Train
model.fit(X, y)

# Run Inference
X_new = torch.tensor([[6, 12], [7, 14]], dtype=torch.float32)
y_pred = model.predict(X_new)
Y_actual = torch.tensor([2*x_1 + 3*x_2 for x_1, x_2 in X_new], dtype=torch.float32)

print("== Results for 2 Features ==")
print(f"Answers: {Y_actual.numpy()}")
print(f"Predictions: {y_pred.squeeze().numpy()}")
print(f"Avg. Error: {torch.mean(torch.abs(y_pred.squeeze() - Y_actual)).item()}")
print()