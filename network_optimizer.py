import numpy as np
from sgd_feedforward import SgdNeuralNetwork
from scg_feedforward import ScgNeuralNetwork
from leapfrog_feedforward import LeapfrogNeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error


def optimize_sgd_number_of_units(train_X, train_Y, input_neurons, hidden_units_start, output_neurons):
    """
    Takes in scaled data, and the number of hidden units to test from
    """
    best_network = SgdNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
    best_network.stochastic_gradient_descent(train_X, train_Y)

    best_performance = np.max(best_network.valid_performance_per_epoch)

    while True:
        hidden_units_start += 1

        avg_best = 0
        for _ in range(1):
            new_network = SgdNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
            new_network.stochastic_gradient_descent(train_X, train_Y)
            if output_neurons == 1:
                avg_best += np.min(new_network.valid_performance_per_epoch)
            else:
                avg_best += np.max(new_network.valid_performance_per_epoch)

        new_best = avg_best / 1

        if output_neurons == 1 and new_best < best_performance or output_neurons > 1 and new_best > best_performance:
            best_performance = new_best
        else:
            break

        print(
            f"No. hidden units {hidden_units_start}: Avg performance = {best_performance:.3f}")

    return hidden_units_start - 1


def optimize_sgd_epochs(train_X, train_Y, input_neurons, hidden_units_start, output_neurons, num_runs=1):
    train_performances = []  # List to store train performance for each run
    valid_performances = []  # List to store validation performance for each run

    for _ in range(num_runs):
        network = SgdNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
        network.stochastic_gradient_descent(train_X, train_Y)

        train_performances.append(network.train_performance_per_epoch)
        valid_performances.append(network.valid_performance_per_epoch)

    # Calculate the average train and validation performances
    avg_train_performance = [sum(epoch[i] for epoch in train_performances) / num_runs for i in range(len(train_performances[0]))]
    avg_valid_performance = [sum(epoch[i] for epoch in valid_performances) / num_runs for i in range(len(valid_performances[0]))]
    optimum_epochs = np.max(np.asarray(avg_valid_performance))

    return optimum_epochs, np.asarray(avg_train_performance), np.asarray(avg_valid_performance)


def optimize_scg_number_of_units(train_X, train_Y, input_neurons, hidden_units_start, output_neurons):
    """
    Takes in scaled data, and the number of hidden units to test from
    """
    best_network = ScgNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
    best_network.scg(train_X, train_Y)

    best_performance = np.max(best_network.valid_performance_per_epoch)

    while True:
        hidden_units_start += 1

        avg_best = 0
        for _ in range(1):
            new_network = SgdNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
            new_network.stochastic_gradient_descent(train_X, train_Y)
            if output_neurons == 1:
                avg_best += np.min(new_network.valid_performance_per_epoch)
            else:
                avg_best += np.max(new_network.valid_performance_per_epoch)

        new_best = avg_best / 1

        if output_neurons == 1 and new_best < best_performance or output_neurons > 1 and new_best > best_performance:
            best_performance = new_best
        else:
            break

        print(
            f"No. hidden units {hidden_units_start}: Avg performance = {best_performance:.3f}")

    return hidden_units_start - 1


def optimize_scg_epochs(train_X, train_Y, input_neurons, hidden_units_start, output_neurons, num_runs=1):
    train_performances = []  # List to store train performance for each run
    valid_performances = []  # List to store validation performance for each run

    for _ in range(num_runs):
        network = ScgNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
        network.scg(train_X, train_Y)

        train_performances.append(network.train_performance_per_epoch)
        valid_performances.append(network.valid_performance_per_epoch)

    # Calculate the average train and validation performances
    avg_train_performance = [sum(epoch[i] for epoch in train_performances) / num_runs for i in range(len(train_performances[0]))]
    avg_valid_performance = [sum(epoch[i] for epoch in valid_performances) / num_runs for i in range(len(valid_performances[0]))]
    optimum_epochs = np.max(np.asarray(avg_valid_performance))

    return optimum_epochs, np.asarray(avg_train_performance), np.asarray(avg_valid_performance)


def optimize_leapfrog_number_of_units(train_X, train_Y, input_neurons, hidden_units_start, output_neurons):
    """
    Takes in scaled data, and the number of hidden units to test from
    """
    best_network = LeapfrogNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
    best_network.leapfrog(train_X, train_Y)

    best_performance = np.max(best_network.valid_performance_per_epoch)

    while True:
        hidden_units_start += 1

        avg_best = 0
        for _ in range(1):
            new_network = SgdNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
            new_network.stochastic_gradient_descent(train_X, train_Y)
            if output_neurons == 1:
                avg_best += np.min(new_network.valid_performance_per_epoch)
            else:
                avg_best += np.max(new_network.valid_performance_per_epoch)

        new_best = avg_best / 1

        if output_neurons == 1 and new_best < best_performance or output_neurons > 1 and new_best > best_performance:
            best_performance = new_best
        else:
            break

        print(
            f"No. hidden units {hidden_units_start}: Avg performance = {best_performance:.3f}")

    return hidden_units_start - 1


def optimize_leapfrog_epochs(train_X, train_Y, input_neurons, hidden_units_start, output_neurons, num_runs=1):
    train_performances = []  # List to store train performance for each run
    valid_performances = []  # List to store validation performance for each run

    for _ in range(num_runs):
        network = LeapfrogNeuralNetwork(input_neurons, hidden_units_start, output_neurons)
        network.leapfrog(train_X, train_Y)

        train_performances.append(network.train_performance_per_epoch)
        valid_performances.append(network.valid_performance_per_epoch)

    # Calculate the average train and validation performances
    avg_train_performance = [sum(epoch[i] for epoch in train_performances) / num_runs for i in range(len(train_performances[0]))]
    avg_valid_performance = [sum(epoch[i] for epoch in valid_performances) / num_runs for i in range(len(valid_performances[0]))]
    optimum_epochs = np.max(np.asarray(avg_valid_performance))

    return optimum_epochs, np.asarray(avg_train_performance), np.asarray(avg_valid_performance)


def evaluate_sgd(input_neurons, hidden_units, output_neurons, train_X, train_Y, test_X, test_Y, optimum_epochs=100, num_runs=3):
    train_perf = []
    test_perf = []

    for _ in range(num_runs):
        network = SgdNeuralNetwork(input_neurons, hidden_units, output_neurons)
        network.stochastic_gradient_descent(train_X, train_Y, epochs=int(optimum_epochs))

        train_predictions = network.feedforward(train_X)
        test_predictions = network.feedforward(test_X)

        if output_neurons == 1:
            train = mean_squared_error(train_predictions, train_Y)
            test = mean_squared_error(test_predictions, test_Y)
        else:
            train = accuracy_score(train_predictions, train_Y)
            test = accuracy_score(test_predictions, test_Y)

        train_perf.append(train)
        test_perf.append(test)

    return np.mean(train_perf), np.std(train_perf), np.mean(test_perf), np.std(test_perf)


def evaluate_scg(input_neurons, hidden_units, output_neurons, train_X, train_Y, test_X, test_Y, optimum_epochs=100,
                 num_runs=3):
    train_perf = []
    test_perf = []

    for _ in range(num_runs):
        network = ScgNeuralNetwork(input_neurons, hidden_units, output_neurons)
        network.scg(train_X, train_Y, epochs=int(optimum_epochs))

        train_predictions = network.feedforward(train_X)
        test_predictions = network.feedforward(test_X)

        if output_neurons == 1:
            train = mean_squared_error(train_predictions, train_Y)
            test = mean_squared_error(test_predictions, test_Y)
        else:
            train = accuracy_score(train_predictions, train_Y)
            test = accuracy_score(test_predictions, test_Y)

        train_perf.append(train)
        test_perf.append(test)

    return np.mean(train_perf), np.std(train_perf), np.mean(test_perf), np.std(test_perf)


def evaluate_leapfrog(input_neurons, hidden_units, output_neurons, train_X, train_Y, test_X, test_Y, optimum_epochs=100,
                 num_runs=3):
    train_perf = []
    test_perf = []

    for _ in range(num_runs):
        network = LeapfrogNeuralNetwork(input_neurons, hidden_units, output_neurons)
        network.leapfrog(train_X, train_Y, epochs=int(optimum_epochs))

        train_predictions = network.feedforward(train_X)
        test_predictions = network.feedforward(test_X)

        if output_neurons == 1:
            train = mean_squared_error(train_predictions, train_Y)
            test = mean_squared_error(test_predictions, test_Y)
        else:
            train = accuracy_score(train_predictions, train_Y)
            test = accuracy_score(test_predictions, test_Y)

        train_perf.append(train)
        test_perf.append(test)

    return np.mean(train_perf), np.std(train_perf), np.mean(test_perf), np.std(test_perf)




def graph_classifier_performance(train_performance, valid_performance, title):
    epochs = range(1, len(train_performance) + 1)

    plt.plot(epochs, train_performance, label='Training', color='blue')
    plt.plot(epochs, valid_performance, label='Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.title(title)
    plt.legend()

    plt.savefig(f'./artefacts/{title.replace(" ", "")}.pdf')
    plt.show()


def graph_regressor_performance(train_performance, valid_performance, title):
    epochs = range(1, len(train_performance) + 1)

    plt.plot(epochs, train_performance, label='Training', color='blue')
    plt.plot(epochs, valid_performance, label='Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    plt.title(title)
    plt.legend()

    plt.savefig(f'./artefacts/{title.replace(" ", "")}.pdf')
    plt.show()
