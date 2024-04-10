import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.03, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def _softmax(self, z):
        # Internal usage only within the LogisticRegression class
        # Underscore in front is more for readability and following convention
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # TODO: Review exp_z contents and why it's axis=1
    
    def _one_hot(self, y):
        # e.g. y = [0, 1, 2, 1]
        num_classes = np.max(y) + 1 # Classes should assumed to be zero-indexed integers of increasing order
        # y.shape = (4,); num_classes = 3
        one_hot_y = np.zeros((y.shape[0], num_classes)) # Initialize y matrix. y.shape = [<num_labels>, <num_classes>]
        one_hot_y[np.arange(y.shape[0]), y] = 1 # Turn element at the class index of each row to 1; one ont-hot per each row vector
        return one_hot_y
    
    def fit(self, X, y):
        num_samples, num_features = X.shape # X.shape = [<num_data>, <num_features>]
        num_classes = np.max(y) + 1

        self.weights = np.zeros((num_features, num_classes)) # weights.shape = [<num_features>, <num_classes>]
        self.bias = np.zeros(num_classes) # One bias for each class; bias.shape = [<num_classes>,]
        
        y_one_hot = self._one_hot(y) # Initialize one_hot matrix

        for _ in range(self.num_iterations):
            # X.shape = [<num_classes>, <num_features>]
            # weights.shape = [<num_features>, <num_classes>]
            # bias.shape = [<num_classes>,]
            z = np.dot(X, self.weights) + self.bias
            logits = self._softmax(z)
            
            # Compute gradients
            gradient_weights = (1 / num_samples) * np.dot(X.T, (logits - y_one_hot))
            gradient_bias = (1 / num_samples) * np.sum(logits - y_one_hot, axis=0) # 
            
            # Gradient Descent
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        logits = self._softmax(z)
        # Recall that logits.shape = [<num_examples>, <num_classes>]
        # So we want a vector of shape [<num_example>,], where each element tells us the
        # most-likely class that this data(one row) belongs to.
        return np.argmax(logits, axis=1) # Find max index over each row vector(i.e. across columns, which is axis=1)            
        

# Make some sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([0, 1, 2, 3])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Run inference
X_test = np.array([[1, 2, 3], [10, 11, 12], [7, 8, 9]])
y_pred = model.predict(X_test)

Y_actual = np.array([0, 3, 2])
print("== Results ==")
print(f"Answers: {Y_actual}")
print(f"Predictions: {y_pred}")
print(f"Avg. Error: {1 - (np.sum(y_pred == Y_actual) / len(Y_actual))}")

        


