import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utilities import one_hot_encode, reverse_one_hot_encode, cross_entropy_loss, linear, softmax, sigmoid


class SgdNeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        fanin_hidden = self.input_neurons
        fanin_output = self.hidden_neurons

        # Initialize biases and weights stochastically
        self.biases = [np.random.randn(hidden_neurons), np.random.randn(output_neurons)]
        self.weights = [
            np.random.uniform(-1 / np.sqrt(fanin_hidden), 1 / np.sqrt(fanin_hidden), (hidden_neurons, input_neurons)),
            np.random.uniform(-1 / np.sqrt(fanin_output), 1 / np.sqrt(fanin_output), (output_neurons, hidden_neurons))
        ]

        if output_neurons == 1: # regression problem
            self.activation_function_hidden = sigmoid
            self.activation_function_output = linear
            self.cost_function = mean_squared_error
            self.performance_measure = mean_squared_error
        else: # classification problem
            self.activation_function_hidden = sigmoid
            self.activation_function_output = softmax
            self.cost_function = cross_entropy_loss
            self.performance_measure = accuracy_score

        self.train_performance_per_epoch = None
        self.valid_performance_per_epoch = None

    def feedforward(self, a):
        z_hidden = np.dot(a, self.weights[0].T) + self.biases[0]
        a_hidden = self.activation_function_hidden(z_hidden)

        z_output = np.dot(a_hidden, self.weights[1].T) + self.biases[1]
        a_output = self.activation_function_output(z_output)

        if self.output_neurons > 1:
            a_output = reverse_one_hot_encode(a_output)

        return a_output

    def stochastic_gradient_descent(self, train_X, train_Y, epochs=300, mini_batch_size=100,
                                    learning_rate=0.001):

        train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

        encoded_Y = one_hot_encode(train_Y) if self.output_neurons > 1 else None

        self.train_performance_per_epoch = np.zeros(epochs)
        self.valid_performance_per_epoch = np.zeros(epochs)

        for epoch in range(epochs):
            permutation = np.random.permutation(train_X.shape[0])
            shuffled_X = train_X[permutation]
            shuffled_Y = train_Y[permutation] if self.output_neurons == 1 else encoded_Y[permutation]

            for i in range(0, train_X.shape[0], mini_batch_size):
                batch_X = shuffled_X[i:i + mini_batch_size]
                batch_Y = shuffled_Y[i:i + mini_batch_size]

                self.update_weights(batch_X, batch_Y, learning_rate)

            # current success measure
            train_predictions = self.feedforward(train_X)
            valid_predictions = self.feedforward(valid_X)

            train_performance = self.performance_measure(train_predictions, train_Y)
            valid_performance = self.performance_measure(valid_predictions, valid_Y)
            self.train_performance_per_epoch[epoch] = train_performance
            self.valid_performance_per_epoch[epoch] = valid_performance

            # if (epoch + 1) % 20 == 0:
            #     print(
            #         f"Epoch {epoch + 1}: Train performance = {train_performance:.3f}, Valid performance = {valid_performance:.3f}")


    def update_weights(self, batch_X, batch_Y, learning_rate):
        # Perform forward pass
        z_hidden = np.dot(batch_X, self.weights[0].T) + self.biases[0]
        a_hidden = self.activation_function_hidden(z_hidden)
        z_output = np.dot(a_hidden, self.weights[1].T) + self.biases[1]
        a_output = self.activation_function_output(z_output)

        # Compute error in output layer
        if self.output_neurons > 1:
            delta_output = a_output - batch_Y
        else:
            delta_output = 2*(a_output - batch_Y.reshape(-1, 1))

        # Compute error in hidden layer
        delta_hidden = np.dot(delta_output, self.weights[1]) * a_hidden * (1 - a_hidden) # where 1 - a_hidden is derivative of sigmoid


        # Compute gradients
        grad_weights_output = np.dot(delta_output.T, a_hidden)
        grad_biases_output = np.sum(delta_output, axis=0)
        grad_weights_hidden = np.dot(delta_hidden.T, batch_X)
        grad_biases_hidden = np.sum(delta_hidden, axis=0)

        # Update weights and biases
        self.weights[1] -= learning_rate * grad_weights_output
        self.biases[1] -= learning_rate * grad_biases_output
        self.weights[0] -= learning_rate * grad_weights_hidden
        self.biases[0] -= learning_rate * grad_biases_hidden


if __name__ == '__main__':
    df = pd.read_csv('./data/mnist.csv').head(5000)
    scalar = StandardScaler()

    # Define features (X) and labels (Y)
    X = scalar.fit_transform(df.iloc[:, 1:].values)
    Y = df.iloc[:, 0].values

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

    nn = SgdNeuralNetwork(784, 8, 10)
    nn.stochastic_gradient_descent(train_X, train_Y)


    train_predictions = nn.feedforward(train_X)
    train_accuracy = accuracy_score(train_predictions, train_Y)
    test_predictions = nn.feedforward(test_X)
    test_accuracy = accuracy_score(test_predictions, test_Y)

    print(train_accuracy)
    print(test_accuracy)
