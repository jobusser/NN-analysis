import numpy as np


def one_hot_encode(labels):
    num_classes = int(np.max(labels) + 1)
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    one_hot_labels[np.arange(num_samples), labels] = 1

    return one_hot_labels


def reverse_one_hot_encode(one_hot_labels):
    if one_hot_labels.ndim == 1:  # Check if the input is 1D
        return np.argmax(one_hot_labels)
    else:
        return np.argmax(one_hot_labels, axis=1)


def cross_entropy_loss(y_true, y_pred): # where y_true is one hot encoded
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


def cross_entropy_derivative(y_true, A2):
    return (y_true / A2) + ((1 - y_true) / (1 - A2))


def linear(z):
    return z

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(z):
    if len(z.shape) == 1:
        shiftx = z - np.max(z)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)
    elif len(z.shape) == 2:
        # Apply softmax along the rows
        shiftx = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_derivative(Z2):
    S = softmax(Z2)
    n = len(S)
    sums = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                sums[i] -= S[j] * S[i]
            else:
                sums[i] += S[i] * (1 - S[i])

    return sums


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    x = sigmoid(z)
    return x * (1 - x)
