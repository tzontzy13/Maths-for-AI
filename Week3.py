import numpy as np
import matplotlib.pyplot as plt

import random
import math

from numpy.random import default_rng
rng = default_rng()


# Generating data following a gaussian distribution


def generate_centered_gaussian(N, cov):
    data = rng.multivariate_normal( (0,0), cov, size = N, check_valid='ignore')
    return data


def plot_points(data_points):
    plt.figure(figsize=(5, 5))
    range_point_min = np.min(data_points)
    range_point_max = np.max(data_points)
    plt.xlim(range_point_min, range_point_max)
    plt.ylim(range_point_min, range_point_max)
    plt.plot(data_points[:, 0], data_points[:, 1], 'g.')
    plt.show()


test_data = generate_centered_gaussian(1000, ((1., 3.), (3., 1.)))
plot_points(test_data)


# Generate more complex data


def generate_gaussian(N, cov, center):
    centered_data = generate_centered_gaussian(N, cov)

    data = centered_data + np.asarray(center)

    return data


test_data = generate_gaussian(1000, ((1, 0), (3, 4)), (3, 4))
plot_points(test_data)


# Data Generator

random.seed(2)


class DataGenerator:

    def __init__(self, distance):

        self.distance = distance

        # Center for first gaussian
        self.center_0 = (0, 0)

        # Center for second gaussian
        theta = random.uniform(0, 2 * math.pi)
        self.center_1 = (distance * math.cos(theta), distance * math.sin(theta))

        # Covariance Matrices
        self.cov_0 = rng.uniform(-1, 1, (2, 2))
        self.cov_1 = rng.uniform(-1, 1, (2, 2))

    def generate_data_class(self, N, label):

        if label == 0:
            X = generate_gaussian(N, self.cov_0, self.center_0)
            y = np.zeros(N)
        else:
            X = generate_gaussian(N, self.cov_1, self.center_1)
            y = np.ones(N)

        return X, y

    def generate_dataset(self, N_0, N_1):

        X_0, y_0 = self.generate_data_class(N_0, 0)
        X_1, y_1 = self.generate_data_class(N_1, 1)

        X = np.concatenate([X_0, X_1])
        y = np.concatenate([y_0, y_1])

        random_indices = np.arange(N_0 + N_1)
        np.random.shuffle(random_indices)
        X = X[random_indices]
        y = y[random_indices]

        return X, y


def plot_points(X, y):
    plt.figure(figsize=(5, 5))
    range_point_min = np.min(X)
    range_point_max = np.max(X)
    plt.xlim(range_point_min, range_point_max)
    plt.ylim(range_point_min, range_point_max)

    plt.scatter(X[:, 0], X[:, 1], marker='.', c=y)

    plt.show()


data = DataGenerator(3)
X, y = data.generate_dataset(100, 100)

plot_points(X, y)

# First Machine Learning algorithm

data = DataGenerator(2)
X_train, y_train = data.generate_dataset(500,500)
X_test, y_test = data.generate_dataset(100,100)

plot_points(X_train, y_train)
plot_points(X_test, y_test)

# Projection


def classify(X, a, b, label_up):
    x_coordinates = X[:, 0]
    y_coordinates = X[:, 1]

    y_pred = (x_coordinates * a + b) > y_coordinates

    if label_up == 0:
        return 1 - 1.0 * y_pred
    else:
        return y_pred * 1.0


def evaluate(y_true, y_pred):
    total_correct = sum(y_true == y_pred)

    return total_correct / len(y_true)


def plot_points_line(X, y, a, b):
    plt.figure(figsize=(5, 5))
    range_point_min = np.min(X)
    range_point_max = np.max(X)
    plt.xlim(range_point_min, range_point_max)
    plt.ylim(range_point_min, range_point_max)

    plt.scatter(X[:, 0], X[:, 1], marker='.', c=y)

    x_line = [range_point_min, range_point_max]
    y_line = [a * x + b for x in x_line]
    plt.plot(x_line, y_line)

    plt.show()


y_pred = classify(X_train, 1, -3, label_up=0)
plot_points_line(X_train, y_pred, 1, -3)

evaluate(y_pred, y_train)


# BSML


class BruteForceStupidML:

    def __init__(self, number_of_steps):

        self.number_of_steps = number_of_steps

        self.current_best_accuracy = 0

        self.best_params = {'a': 0, 'b': 0, 'label_up': 0}

        self.results = [(0, self.current_best_accuracy)]

    def train(self, X_train, y_train):

        for step in range(1, self.number_of_steps + 1):

            a_random = rng.uniform(-10, 10)
            b_random = rng.uniform(-10, 10)
            label_up_random = rng.choice([0, 1])

            y_pred = classify(X_train, a_random, b_random, label_up_random)

            result = evaluate(y_pred, y_train)

            if result > self.current_best_accuracy:
                self.best_params['a'] = a_random
                self.best_params['b'] = b_random
                self.best_params['label_up'] = label_up_random

                self.current_best_accuracy = result

            self.results.append((step, self.current_best_accuracy))

    def plot_training_curve(self):

        x_results = [r[0] for r in self.results]
        y_results = [r[1] for r in self.results]

        plt.plot(x_results, y_results, '-b')

        plt.show()

    def test(self, X_test, y_test):

        a = self.best_params['a']
        b = self.best_params['b']
        label_up = self.best_params['label_up']

        y_pred = classify(X_train, a, b, label_up)

        return evaluate(y_pred, y_train)


bfsml = BruteForceStupidML(100)
bfsml.train(X_train, y_train)
bfsml.plot_training_curve()
