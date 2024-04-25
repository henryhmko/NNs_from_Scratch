import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes, learning_rate=0.003, num_iterations=10000):
        super(LogisticRegression, self).__init__()
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X, y):
        num_samples, num_features = X.shape # X.shape = [num_data, num_features]
        criterion = nn.CrossEntropyLoss() # Use the cross entropy loss for classification of multiiple classes
        optimizer = torch.optim.SGD(self.parameters(), self.learning_rate)

        for _ in range(self.num_iterations):
            logits = self.forward(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1) # Return the class index holding the largest probability in each row(i.e. dim=1)

# Make some sample data
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32)
y = torch.tensor([    0,         1,         2,           3])

# Create model
num_features = X.shape[1]
num_classes = torch.max(y) + 1 # Since it's 0-indexed
model = LogisticRegression(num_features, num_classes)

# Train model
model.fit(X, y)

# Run inference
X_test = torch.tensor([[1, 2, 3], [10, 11, 12], [7, 8, 9]], dtype=torch.float32)
y_pred = model.predict(X_test)
Y_actual = torch.tensor([0, 3, 2])

print("== Results ==")
print(f"Answers: {Y_actual.numpy()}")
print(f"Predictions: {y_pred.numpy()}")
print(f"Avg. Error: {1 - (torch.sum(y_pred == Y_actual) / len(Y_actual)).item()}")