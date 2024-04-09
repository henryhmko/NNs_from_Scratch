import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.03, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape # X is a mxn matrix where m is num_samples and n is num_features
        self.weights = np.zeros(num_features) # Initialize weights; num_weights == num_features
        self.bias = 0 # Initialize bias to 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias # Since y_hat = AX + b
            
            # In practice, we would use 1/num_samples since it's simpler computation while preserving the direction of the gradient.
            # Intuitively, 2/num_samples would tell us not only the direction of steepest ascent, but also the magnitude.
            # The latter is determined by our learning rate, which is a hyperparameter
            
            # I left it as 2/num_samples to avoid confusions for its derivation
            dw = (2 / num_samples) * np.dot(X.T, (y_predicted - y)) # See notes for derivation
            db = (2 / num_samples) * np.sum(y_predicted - y) # See notes for derivation

            # Recalibrate weights & bias by doing gradient descent
            # i.e. step in the opposite direction of the gradient with the chosen step size(learning rate)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
    def predict(self, X):
        # Weights are fixed now, so simply run y_hat = WX + b
        # return np.dot(self.weights, X) + self.bias
        return np.dot(X, self.weights) + self.bias
    

# Run tests

# Make some sample data with only one feature. 
# i.e. x_i.shape == (1,)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([x*10 for x in range(1,6)]) # Expected output is 10*(input)

# Create model
model = LinearRegression()

# Train
model.fit(X, y)

# Run Inference
X_new = np.array([[6], [7]])
y_pred = model.predict(X_new)

Y_actual = [60, 70]

print(f"Predictions: {y_pred}")
print(f"Error: {np.mean(np.abs(y_pred - Y_actual))}")