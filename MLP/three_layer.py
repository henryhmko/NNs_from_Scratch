import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, learning_rate=0.0003, num_iterations=8000):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self.weights_hidden_1 = np.random.rand(input_size, hidden_size_1)
        self.bias_hidden_1 = np.zeros(self.hidden_size_1)

        self.weights_hidden_2 = np.random.rand(hidden_size_1, hidden_size_2)
        self.bias_hidden_2 = np.zeros(self.hidden_size_2)

        self.output_layer_weights = np.random.rand(hidden_size_2, output_size)
        self.output_layer_bias = np.zeros(self.output_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _forward(self, x):
        self.hidden_layer_1 = self._relu(x @ self.weights_hidden_1 + self.bias_hidden_1)
        self.hidden_layer_2 = self._relu(self.hidden_layer_1 @ self.weights_hidden_2 + self.bias_hidden_2)
        self.output_layer = self._relu(self.hidden_layer_2 @ self.output_layer_weights + self.output_layer_bias)
        return self.output_layer

    def _backward(self, x, y, output):
        error_output = output - y.reshape(-1, 1)
        # Apply chain rule
        grad_output = error_output * self._relu_derivative(output)
        # NOTE: # Technically, we should take an average of the summed gradients instead, but the learning rate accounts for this scaling.
        # Otherwise, we would be scaling the learning rate by (1 / batch_size), which will disturb training
        # One more note: ideally we should make hyperparameters independent of each other; changing one should not change the other significantly without us knowing
        bias_output = np.sum(grad_output, axis=0)
        
        error_hidden_2 = grad_output @ self.output_layer_weights.T
        grad_hidden_2 = error_hidden_2 * self._relu_derivative(self.hidden_layer_2)
        bias_hidden_2 = np.sum(grad_hidden_2, axis=0)

        error_hidden_1 = grad_hidden_2 @ self.weights_hidden_2.T
        grad_hidden_1 = error_hidden_1 * self._relu_derivative(self.hidden_layer_1)
        bias_hidden_1 = np.sum(grad_hidden_1, axis=0)

        # Update weights and biases
        self.weights_hidden_1 -= self.learning_rate * (x.T @ grad_hidden_1)
        self.bias_hidden_1 -= self.learning_rate * bias_hidden_1

        self.weights_hidden_2 -= self.learning_rate * (self.hidden_layer_1.T @ grad_hidden_2)
        self.bias_hidden_2 -= self.learning_rate * bias_hidden_2

        self.output_layer_weights -= self.learning_rate * (self.hidden_layer_2.T @ grad_output)
        self.output_layer_bias -= self.learning_rate * bias_output

    def fit(self, X, y):
        for _ in range(self.num_iterations):
            output = self._forward(X)
            self._backward(X, y, output)
    
    def predict(self, X):
        return self._forward(X)

# Make some sample data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=np.float32)
y = np.array([2*x_1 + 3*x_2 for x_1, x_2 in X], dtype=np.float32).reshape(-1, 1)

# Create model
num_features = X.shape[1]
model = MLP(input_size=num_features, hidden_size_1=num_features*2, hidden_size_2=num_features*4, output_size=1)

# Train model
model.fit(X, y)

# Run inference
X_new = np.array([[6, 12], [7, 14]], dtype=np.float32)
y_pred = model.predict(X_new)
Y_actual = np.array([2*x_1 + 3*x_2 for x_1, x_2 in X_new], dtype=np.float32).reshape(-1, 1)

print("== Results ==")
print(f"Answers: {Y_actual.flatten()}")
print(f"Predictions: {y_pred.flatten()}")
print(f"Avg. Error: {np.mean(np.abs(y_pred.flatten() - Y_actual.flatten()))}")
print()