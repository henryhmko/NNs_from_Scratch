import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, num_iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        # Inititalize hidden weights as a matrix of (input_size x hidden_size)
        self.weights_hidden = np.random.randn(input_size, hidden_size) # Recall that .randn outputs values between [-1, 1] vs .rand outputs [0, 1)
        # Initialize hidden bias as a vector of zeros with number of elements equal to hidden_size
        self.bias_hidden = np.zeros(hidden_size)
        # Inititalize output weights as matrix of (hidden_size x output_size) 
        self.weights_output = np.random.randn(hidden_size, output_size)
        # Final bias has output_size number of elements this time
        self.bias_output = np.zeros(output_size)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        # Check derivation in the notes
        # Note that this is not the direct derivative you would get if you calculate it yourself. 
        # We used a simplified version of the derivative and the derivation in included in the notes.
        return x * (1 - x)
    
    def _forward(self, X):
        hidden_layer_before_sigmoid = np.dot(X, self.weights_hidden) + self.bias_hidden # X @ w_hidden + bias_hidden
        self.hidden_layer = self._sigmoid(hidden_layer_before_sigmoid) # Apply nonlinearity
        
        # Repeat (num_layer - 1) times.
        # We will go straight to output_layer since we only have 1 layer
        output_layer_before_sigmoid = np.dot(self.hidden_layer, self.weights_output) + self.bias_output
        self.output_layer = self._sigmoid(output_layer_before_sigmoid)
        
    def _backward(self, X, y, output):
        error_output = y - output # Error of output 
        gradient_output = error_output * self._sigmoid_derivative(output) # Gradient of output
        
        error_hidden = np.dot(gradient_output, self.weights_output.T)
        gradient_hidden = error_hidden * self._sigmoid_derivative(self.hidden_layer)
    


        
















class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, num_iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _forward(self, X):
        self.hidden_layer = self._sigmoid(np.dot(X, self.weights_hidden) + self.bias_hidden)
        self.output_layer = self._sigmoid(np.dot(self.hidden_layer, self.weights_output) + self.bias_output)

    def _backward(self, X, y, output):
        error_output = y - output
        gradient_output = error_output * self._sigmoid_derivative(output)

        error_hidden = np.dot(gradient_output, self.weights_output.T)
        gradient_hidden = error_hidden * self._sigmoid_derivative(self.hidden_layer)

        self.weights_output += self.learning_rate * np.dot(self.hidden_layer.T, gradient_output)
        self.bias_output += self.learning_rate * np.sum(gradient_output, axis=0)
        self.weights_hidden += self.learning_rate * np.dot(X.T, gradient_hidden)
        self.bias_hidden += self.learning_rate * np.sum(gradient_hidden, axis=0)

    def fit(self, X, y):
        for _ in range(self.num_iterations):
            self._forward(X)
            self._backward(X, y, self.output_layer)

    def predict(self, X):
        self._forward(X)
        return self.output_layer


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=4, output_size=1)
mlp.fit(X, y)

test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = mlp.predict(test_input)

print("Predictions:")
print(predictions)




import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, num_iterations=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros(output_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _forward(self, X):
        self.hidden_layer = self._relu(np.dot(X, self.weights_hidden) + self.bias_hidden)
        self.output_layer = np.dot(self.hidden_layer, self.weights_output) + self.bias_output

    def _backward(self, X, y, output):
        error_output = output - y
        gradient_output = error_output

        error_hidden = np.dot(gradient_output, self.weights_output.T)
        gradient_hidden = error_hidden * self._relu_derivative(self.hidden_layer)

        self.weights_output -= self.learning_rate * np.dot(self.hidden_layer.T, gradient_output)
        self.bias_output -= self.learning_rate * np.sum(gradient_output, axis=0)
        self.weights_hidden -= self.learning_rate * np.dot(X.T, gradient_hidden)
        self.bias_hidden -= self.learning_rate * np.sum(gradient_hidden, axis=0)

    def fit(self, X, y):
        for _ in range(self.num_iterations):
            self._forward(X)
            self._backward(X, y, self.output_layer)

    def predict(self, X):
        self._forward(X)
        return self.output_layer


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=4, output_size=1)
mlp.fit(X, y)

test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = mlp.predict(test_input)

print("Predictions:")
print(predictions)