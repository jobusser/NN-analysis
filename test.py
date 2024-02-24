import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utilities import one_hot_encode, reverse_one_hot_encode, cross_entropy_loss, linear, softmax, sigmoid, sigmoid_derivative, softmax_derivative, cross_entropy_derivative


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


        self.activation_function_hidden = sigmoid
        self.activation_function_input_derivative = sigmoid_derivative
        self.activation_function_output = softmax
        self.activation_function_output_derivative = softmax_derivative
        self.cost_function = cross_entropy_loss
        self.cost_derivative = cross_entropy_derivative
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

        encoded_Y = one_hot_encode(train_Y)

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

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch + 1}: Train performance = {train_performance:.3f}, Valid performance = {valid_performance:.3f}")


    def update_weights(self, batch_X, batch_Y, learning_rate):
        grad_b_hidden = np.zeros(self.biases[0].shape)
        grad_b_output = np.zeros(self.biases[1].shape)
        grad_w_hidden = np.zeros(self.weights[0].shape)
        grad_w_output = np.zeros(self.weights[1].shape)

        for x,y in zip(batch_X, batch_Y):
            # forward pass
            z_hidden = np.dot(x, self.weights[0].T) + self.biases[0]
            a_hidden = self.activation_function_hidden(z_hidden)
            z_output = np.dot(a_hidden, self.weights[1].T) + self.biases[1]
            a_output = self.activation_function_output(z_output)

            # backwards pass
            dC_dA2 = self.cost_derivative(y, a_output)
            dA2_dZ2 = self.activation_function_output_derivative(z_output)
            dZ2_dW2 = a_output
            delta_output = np.dot(dC_dA2 * dA2_dZ2, dZ2_dW2.T)  # (10, 8)
            print('wooooo')
            print(delta_output)
            print(delta_output.shape)

            # Calculate error in the hidden layer
            delta_hidden = np.dot(delta_output, self.weights[1]) * self.activation_function_input_derivative(z_hidden)

            # Accumulate gradients
            grad_b_output += delta_output
            grad_w_output += np.outer(delta_output, a_hidden)
            grad_b_hidden += delta_hidden
            grad_w_hidden += np.outer(delta_hidden, x)

        # Update weights and biases
        self.weights[1] -= (learning_rate / len(batch_X)) * grad_w_output
        self.biases[1] -= (learning_rate / len(batch_X)) * grad_b_output
        self.weights[0] -= (learning_rate / len(batch_X)) * grad_w_hidden
        self.biases[0] -= (learning_rate / len(batch_X)) * grad_b_hidden




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
