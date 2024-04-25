import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0003, num_iterations=3000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # Inititalize hidden weights as a matrix of (input_size x hidden_size)
        # NOTE: initializing weights with .randn leads to failed training more than ~50% of the time for this code
        # TODO: Must initialize weights with NON-NEGATIVE rational numbers
        self.weights_hidden = np.random.rand(input_size, hidden_size) # Recall that .randn outputs values between [-1, 1] vs .rand outputs [0, 1)
        # Initialize hidden bias as a vector of zeros with number of elements equal to hidden_size
        self.bias_hidden = np.zeros(hidden_size)
        # Inititalize output weights as matrix of (hidden_size x output_size)
        self.weights_output = np.random.rand(hidden_size, output_size)
        # Final bias has output_size number of elements this time
        self.bias_output = np.zeros(output_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float) # In ML, we pretend relu is differentiable by saying the derivative at x=0 is 0
    
    def _forward(self, X):
        self.hidden_layer = self._relu(X @ self.weights_hidden + self.bias_hidden)
        self.output_layer = self._relu(self.hidden_layer @ self.weights_output + self.bias_output)
        return self.output_layer

    def _backward(self, x, y, output):
        error_output = output - y.reshape(-1, 1) # Practice more reshaping - I should know which dimensions these are
        # Compute gradient of the loss w.r.t the output layer
        # calculated by multiplying the error at the output layer with the derivative of the ReLU activation function
        gradient_output = error_output * self._relu_derivative(output)
        error_hidden = gradient_output @ self.weights_output.T
        gradient_hidden = error_hidden * self._relu_derivative(self.hidden_layer)
        
        # Update weights and biases of output layer
        self.weights_output -= self.learning_rate * (self.hidden_layer.T @ gradient_output) 
        self.bias_output -= self.learning_rate * np.sum(gradient_output, axis=0) 
        self.weights_hidden -= self.learning_rate * (x.T @ gradient_hidden)
        self.bias_hidden -= self.learning_rate * np.sum(gradient_hidden, axis=0)
    
    def fit(self, X, y):
        for _ in range(self.num_iterations):
            output = self._forward(X)
            self._backward(X, y, output)
    
    def predict(self, X):
        return self._forward(X)
    
# Make some sample data
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]], dtype=np.float32)
y = np.array([2*x_1 + 3*x_2 for x_1, x_2 in X], dtype=np.float32)

# Create model
num_features = X.shape[1]
model = MLP(input_size=num_features, hidden_size = num_features*2, output_size=1)

# Train model
model.fit(X, y)

# Run inference
X_new = np.array([[6, 12], [7, 14]], dtype=np.float32)
y_pred = model.predict(X_new)
Y_actual = np.array([2*x_1 + 3*x_2 for x_1, x_2 in X_new], dtype=np.float32)

print("== Results ==")
print(f"Answers: {Y_actual}")
print(f"Predictions: {y_pred.squeeze()}")
print(f"Avg. Error: {np.mean(np.abs(y_pred.squeeze() - Y_actual)).item()}")
print()