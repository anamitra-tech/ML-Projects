import numpy as np

# -----------------------------
# Activation functions
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # a is sigmoid(z)
    return a * (1 - a)

# -----------------------------
# Neural Network
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        np.random.seed(42)

        # Weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward_pass(self, X):
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, Y, Y_hat):
        # Mean Squared Error
        return np.mean((Y - Y_hat) ** 2)

    def backward_pass(self, X, Y, lr=0.1):
        m = X.shape[0]

        # -------- Output layer gradients --------
        dA2 = self.A2 - Y
        dZ2 = dA2 * sigmoid_derivative(self.A2)

        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # -------- Hidden layer gradients --------
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)

        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # -------- Update --------
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, Y, epochs=1000, lr=0.1):
        for i in range(epochs):
            Y_hat = self.forward_pass(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward_pass(X, Y, lr)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.6f}")
