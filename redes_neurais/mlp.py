import numpy as np
from typing import Literal

# https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d
# + gpt
class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size,epochs=1000, learning_rate=0.01,activation_func:Literal["relu","sigmoid","tanh"]="relu"):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation_func = activation_func

        sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []

        # Xavier initialization
        for i in range(1, self.num_layers + 1):
            self.weights.append(
                np.random.randn(sizes[i], sizes[i-1]) / np.sqrt(sizes[i-1])
            )
            self.biases.append(np.zeros((sizes[i], 1)))

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def gradient_relu(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        # Evita overflow numérico
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def gradient_sigmoid(self, Z):
        S = self.sigmoid(Z)
        return S * (1 - S)
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def gradient_tanh(self, Z):
        return 1 - np.tanh(Z) ** 2

    def activation(self, Z):
        if self.activation_func == "relu":
            return self.relu(Z)

        elif self.activation_func == "sigmoid":
            return self.sigmoid(Z)

        elif self.activation_func == "tanh":
            return self.tanh(Z)

    def activation_gradient(self, Z):
        if self.activation_func == "relu":
            return self.gradient_relu(Z)

        elif self.activation_func == "sigmoid":
            return self.gradient_sigmoid(Z)

        elif self.activation_func == "tanh":
            return self.gradient_tanh(Z)

    def forward(self, X):
        # X: (n_samples, n_features)
        self.activations = [X]
        self.z = []

        for i in range(self.num_layers):
            z = np.dot(self.activations[i], self.weights[i].T) + self.biases[i].T

            self.z.append(z)

            if i < self.num_layers - 1:
                a = self.activation(z)
            else:
                a = z  # Linear output

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]

        gradients = []

        # Derivative of MSE
        dZ = 2 * (self.activations[-1] - y) / (m * self.output_size)

        for i in range(self.num_layers - 1, -1, -1):

            # Weight gradient
            dW = np.dot(dZ.T, self.activations[i])

            # Bias gradient
            db = np.sum(dZ, axis=0, keepdims=True).T

            gradients.append((dW, db))

            if i > 0:
                dA = np.dot(dZ, self.weights[i])

                dZ = dA * self.activation_gradient(self.z[i-1])

        return gradients[::-1]

    def update_parameters(self, gradients):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * gradients[i][0]
            self.biases[i] -= self.learning_rate * gradients[i][1]

    def fit(self, X, y, verbose=True):

        # X = np.asarray(X, dtype=float)
        # y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[1] != self.input_size:
            raise ValueError("X has an incorrect number of features.")

        if y.shape[1] != self.output_size:
            raise ValueError("y has an incorrect number of outputs.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.loss_history = []

        for epoch in range(self.epochs):

            # Forward pass
            outputs = self.forward(X)

            # Calculate MSE
            loss = np.mean((outputs - y) ** 2)
            self.loss_history.append(loss)

            # Backward pass
            gradients = self.backward(X, y)

            # Update parameters
            self.update_parameters(gradients)

            if verbose and (epoch + 1) % self.epochs == 0:
                print(f"Epoch {epoch+1} - Loss: {loss:.6f}")

        return self

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.input_size:
            raise ValueError("X has an incorrect number of features.")

        return self.forward(X)

if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44990_brazilian_houses.csv",delimiter=",",skiprows=1,usecols=(2,3,4,5,9,13))

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    fds = MultilayerPerceptron(
        X.shape[1],
        [3,3,3,3],
        1,
        epochs=125,
        learning_rate=0.1
    )

    fds.fit(X,y)
    print(fds.predict(X))

