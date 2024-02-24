import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from utilities import one_hot_encode, reverse_one_hot_encode, cross_entropy_loss, linear, softmax, sigmoid


class ScgNeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        fanin_hidden = self.input_neurons
        fanin_output = self.hidden_neurons

        hidden = np.random.uniform(-1 / np.sqrt(fanin_hidden), 1 / np.sqrt(fanin_hidden),
                                   (hidden_neurons * (input_neurons + 1),))
        output = np.random.uniform(-1 / np.sqrt(fanin_output), 1 / np.sqrt(fanin_output),
                                   (output_neurons * (hidden_neurons + 1),))

        self.weights = np.concatenate([hidden, output])

        if output_neurons == 1:  # regression problem
            self.activation_function_hidden = sigmoid
            self.activation_function_output = linear
            self.cost_function = mean_squared_error
            self.performance_measure = mean_squared_error
        else:  # classification problem
            self.activation_function_hidden = sigmoid
            self.activation_function_output = softmax
            self.cost_function = cross_entropy_loss
            self.performance_measure = accuracy_score

        self.train_performance_per_epoch = None
        self.valid_performance_per_epoch = None

    def feedforward_matrices(self, flat_weights):
        hidden_bias_idx = self.input_neurons * self.hidden_neurons
        output_weights_idx = (self.input_neurons + 1) * self.hidden_neurons
        output_bias_idx = len(self.weights) - self.output_neurons

        hidden_weights = np.reshape(flat_weights[:hidden_bias_idx], (self.hidden_neurons, self.input_neurons))
        hidden_biases = flat_weights[hidden_bias_idx:output_weights_idx]
        output_weights = np.reshape(flat_weights[output_weights_idx:output_bias_idx],
                                    (self.output_neurons, self.hidden_neurons))
        output_biases = flat_weights[output_bias_idx:]

        return hidden_weights, hidden_biases, output_weights, output_biases

    def feedforward(self, a):
        hidden_weights, hidden_biases, output_weights, output_biases = self.feedforward_matrices(self.weights)

        z_hidden = np.dot(a, hidden_weights.T) + hidden_biases
        a_hidden = self.activation_function_hidden(z_hidden)
        z_output = np.dot(a_hidden, output_weights.T) + output_biases
        a_output = self.activation_function_output(z_output)

        if self.output_neurons > 1:
            a_output = reverse_one_hot_encode(a_output)

        return a_output

    def calculate_error(self, train_X, train_Y, flat_weights):
        hidden_weights, hidden_biases, output_weights, output_biases = self.feedforward_matrices(flat_weights)
        z_hidden = np.dot(train_X, hidden_weights.T) + hidden_biases
        a_hidden = self.activation_function_hidden(z_hidden)
        z_output = np.dot(a_hidden, output_weights.T) + output_biases
        a_output = self.activation_function_output(z_output)

        # if self.output_neurons > 1:
        #     print('approximate accuracy:',
        #           self.performance_measure(reverse_one_hot_encode(a_output), reverse_one_hot_encode(train_Y)),
        #           '\tapproximate CEL', self.cost_function(a_output, train_Y))
        # else:
        #     print('approximate MSE:', self.performance_measure(a_output, train_Y))

        return self.cost_function(a_output, train_Y)

    def calculate_error_derivative(self, train_X, train_Y, flat_weights):
        hidden_weights, hidden_biases, output_weights, output_biases = self.feedforward_matrices(flat_weights)

        z_hidden = np.dot(train_X, hidden_weights.T) + hidden_biases
        a_hidden = self.activation_function_hidden(z_hidden)
        z_output = np.dot(a_hidden, output_weights.T) + output_biases
        a_output = self.activation_function_output(z_output)

        # Compute error in output layer
        if self.output_neurons > 1:  # classification
            delta_output = a_output - train_Y
        else:  # regression
            delta_output = a_output - train_Y.reshape(-1, 1)
        delta_hidden = np.dot(delta_output, output_weights) * a_hidden * (1 - a_hidden)

        grad_weights_output = np.dot(delta_output.T, a_hidden).flatten()
        grad_biases_output = np.sum(delta_output, axis=0)
        grad_weights_hidden = np.dot(delta_hidden.T, train_X).flatten()
        grad_biases_hidden = np.sum(delta_hidden, axis=0)

        return np.concatenate([grad_weights_hidden, grad_biases_hidden, grad_weights_output, grad_biases_output])

    def scg(self, train_X, train_Y, sigma=0.0000000001, lam_0=0.0000000000000001, lam_line=0, epochs=-1):
        train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

        if self.output_neurons > 1:
            train_Y = one_hot_encode(train_Y)
            valid_Y = one_hot_encode(valid_Y)

        n_w = len(self.weights)
        max_epochs_without_improvement = max([int(n_w / 50), 300])

        self.train_performance_per_epoch = []
        self.valid_performance_per_epoch = []

        p = -self.calculate_error_derivative(train_X, train_Y, self.weights)
        r = np.copy(p)
        success = True
        delta = None
        lam = lam_0

        while True:
            for t in range(n_w + 2):

                train_predictions = self.feedforward(train_X)
                valid_predictions = self.feedforward(valid_X)

                if self.output_neurons == 1:
                    train_performance = self.performance_measure(train_predictions, train_Y)
                    valid_performance = self.performance_measure(valid_predictions, valid_Y)

                    did_improve = len(self.valid_performance_per_epoch) > max_epochs_without_improvement + 1 and \
                                  valid_performance >= self.valid_performance_per_epoch[
                                      -(max_epochs_without_improvement + 1)]
                else:
                    train_performance = self.performance_measure(train_predictions, reverse_one_hot_encode(train_Y))
                    valid_performance = self.performance_measure(valid_predictions, reverse_one_hot_encode(valid_Y))

                    did_improve = len(self.valid_performance_per_epoch) > max_epochs_without_improvement + 1 and \
                                  valid_performance <= self.valid_performance_per_epoch[
                                      -(max_epochs_without_improvement + 1)]

                self.train_performance_per_epoch.append(train_performance)
                self.valid_performance_per_epoch.append(valid_performance)

                if len(self.valid_performance_per_epoch) > max_epochs_without_improvement + 1 and not did_improve:
                    return

                if success:
                    delta = self.second_order_info(train_X, train_Y, sigma, p)
                delta = self.scale_delta(delta, lam, lam_line, p)

                if delta <= 0:
                    lam, lam_line, delta = self.hessian_positive_definite(lam, delta, p)

                mu, step_size = self.calc_step_size(p, r, delta)
                comparison = self.comparison_parameter(train_X, train_Y, delta, step_size, p, mu)

                if comparison >= 0:
                    # adjust weights
                    self.weights = self.weights + step_size * p
                    r_new = -self.calculate_error_derivative(train_X, train_Y, self.weights)
                    lam_line = 0
                    success = True

                    if t % n_w == 0:
                        p = r_new
                        lam_line = 0
                        lam = lam_0
                        success = True
                        break
                    else:
                        p = self.new_conjugate_direction(r, r_new, mu, p)
                        r = r_new

                    if comparison >= 0.75:
                        lam = 0.25 * lam

                else:
                    lam_line = lam
                    success = False

                if comparison < 0.25:
                    # lam = 4*lam
                    lam = lam + (delta * (1 - comparison)) / (np.linalg.norm(p) ** 2)

                if not np.allclose(r, 0, atol=1e-2):
                    continue
                else:
                    return

    def second_order_info(self, train_X, train_Y, sigma, p):
        norm_p = np.linalg.norm(p)
        sigma_t = sigma / norm_p

        # Calculate step_size
        step_error = self.calculate_error_derivative(train_X, train_Y, self.weights + sigma_t * p)
        w_error = self.calculate_error_derivative(train_X, train_Y, self.weights)

        step_size = (step_error - w_error) / sigma_t

        delta = np.dot(p, step_size)

        return delta

    def scale_delta(self, delta, lam, lam_line, p):
        delta = delta + (lam - lam_line) * np.linalg.norm(p) ** 2
        return delta

    def hessian_positive_definite(self, lam, delta, p):
        p_norm_sqr = np.linalg.norm(p) ** 2
        lam_line = 2 * (lam - delta / p_norm_sqr)
        delta = -delta + lam * p_norm_sqr
        lam = lam_line
        return lam, lam_line, delta

    def calc_step_size(self, p, r, delta):
        mu = np.dot(p, r)
        step_size = mu / delta
        return mu, step_size

    def comparison_parameter(self, train_X, train_Y, delta, step_size, p, mu):
        current_error = self.calculate_error(train_X, train_Y, self.weights)
        step_error = self.calculate_error(train_X, train_Y, self.weights + step_size * p)

        delta_t = 2 * delta * (current_error - step_error) / (mu ** 2)
        return delta_t

    def new_conjugate_direction(self, r_old, r_new, mu, p):
        beta = ((np.linalg.norm(r_new) ** 2) - np.dot(r_new, r_old)) / mu
        return r_new + beta * p


if __name__ == '__main__':
    # df = pd.read_csv('./data/mnist.csv').head(5000)
    # scalar = StandardScaler()
    #
    # # Define features (X) and labels (Y)
    # X = scalar.fit_transform(df.iloc[:, 1:].values)
    # Y = df.iloc[:, 0].values
    #
    # train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
    #
    # nn = ScgNeuralNetwork(784, 8, 10)
    # nn.scg(train_X, train_Y)
    #
    # train_predictions = nn.feedforward(train_X)
    # train_accuracy = accuracy_score(train_predictions, train_Y)
    # test_predictions = nn.feedforward(test_X)
    # test_accuracy = accuracy_score(test_predictions, test_Y)
    #
    # print(train_accuracy)
    # print(test_accuracy)

    df = pd.read_csv('./data/winequality-red.csv')

    # Define features (X) and labels (Y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    Y = df.iloc[:, -1].values  # The last column

    # Standardize the features
    scalers = [StandardScaler() for _ in range(X.shape[1])]
    X = np.hstack([scaler.fit_transform(X[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers)])

    # Split the data into training and testing sets (80%/20%)
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


    nn = ScgNeuralNetwork(11, 5, 1)
    nn.scg(train_X, train_Y)

    train_predictions = nn.feedforward(train_X)
    train_accuracy = mean_squared_error(train_predictions, train_Y)
    test_predictions = nn.feedforward(test_X)
    test_accuracy = mean_squared_error(test_predictions, test_Y)

    print(train_accuracy)
    print(test_accuracy)
