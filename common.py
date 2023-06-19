import numpy
import torch
import time
import matplotlib.pyplot as plt

# Generate a grid with dimension of [n] * [n]
def generate_dataset_x(n):
    step = 2 / (n - 1)
    result = torch.zeros(n * n, 2)
    for y in range(n):
        for x in range(n):
            result[y * n + x] = torch.tensor([x * step - 1, y * step - 1])
    return result


# Generate a [n] * [n] label with [i]th marked as 1
def generate_dataset_y(n, i):
    result = torch.zeros(n * n)
    result[i] = 1
    return result.view(n * n, 1)


# %%
class TestData:
    def __init__(self, n):
        self.n = n
        self.X = generate_dataset_x(n)
        self.Ys = [generate_dataset_y(n, i) for i in range(n * n)]


class TestParameter:
    def __init__(self, hidden_layers, layer_size, activation_func, max_epoch, lr, betas, eps, weight_decay, test_data):
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.activation_func = activation_func
        self.max_epoch = max_epoch

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.test_data = test_data


# %%

# Plotting function
class TestResult:
    def __init__(self, i, X, Y, n):
        self.i = i
        self.X = X
        self.Y = Y
        self.epoch_loss_x = numpy.ndarray(n)
        self.epoch_loss_y = numpy.ndarray(n)
        self.y_pred = None

    def plot_epoch_loss(self):
        plt.plot(self.epoch_loss_x, self.epoch_loss_y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def plot_pred_grid(self):
        X = self.X
        y_pred = self.y_pred

        f, ax = plt.subplots(figsize=(5, 4))

        ax.plot(X[self.i, 0], X[self.i, 1], marker="o", markersize=20, markeredgecolor="black", markerfacecolor="none")
        p = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="cool")
        f.colorbar(p)

        plt.show()


def plot_score(X, score):
    X = X.cpu()

    f, ax = plt.subplots(figsize=(9, 8))

    p = ax.scatter(X[:, 0], X[:, 1], c=score, cmap="cool")
    f.colorbar(p)

    plt.show()


# %%
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, layer_size, activation_func=lambda: None):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size

        self.sequence = torch.nn.Sequential(torch.nn.Linear(input_size, layer_size))

        for i in range(hidden_layers):
            activation = activation_func()
            if activation is not None:
                self.sequence.append(activation)
            self.sequence.append(torch.nn.Linear(layer_size, layer_size if i < hidden_layers - 1 else output_size))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.sequence(x)


def run_test_with_network(network, optimizer, loss_func, X, Y, i, max_epoch):
    # Train network for [max_epoch]
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        y_pred = network(X)
        loss = loss_func(y_pred, Y).mean()
        loss.backward()
        optimizer.step()

    # Calculate sensitivity after training
    with torch.no_grad():
        network.eval()
        y_pred = network(X)
        y_pred = torch.abs(y_pred)
        return y_pred[i] / y_pred.sum()


def run_test_with_network_full_result(network, optimizer, loss_func, X, Y, i, max_epoch):
    result = TestResult(i, X, Y, max_epoch)

    for epoch in range(max_epoch):
        optimizer.zero_grad()
        y_pred = network(X)
        loss = loss_func(y_pred, Y).mean()
        loss.backward()
        optimizer.step()

        result.epoch_loss_x[epoch] = epoch + 1
        result.epoch_loss_y[epoch] = loss.item()

    with torch.no_grad():
        network.eval()
        result.y_pred = network(X)
        result.y_pred = torch.maximum(result.y_pred, torch.zeros_like(result.y_pred))

    return result


def run_test_for_single_y(parameter: TestParameter, i):
    network = NeuralNetwork(2, 1, parameter.hidden_layers, parameter.layer_size, parameter.activation_func)
    optimizer = torch.optim.Adam(network.parameters(), lr=parameter.lr, betas=parameter.betas, eps=parameter.eps,
                                 weight_decay=parameter.weight_decay)

    def loss(y_pred, y):
        return ((y_pred - y) ** 2) * (y * len(y) + (1 - y))

    return run_test_with_network(network, optimizer, loss, parameter.test_data.X, parameter.test_data.Ys[i], i,
                                 parameter.max_epoch)


def run_test_for_all_y(parameter):
    result = numpy.zeros(parameter.test_data.n ** 2)
    total = len(result)

    for i in range(total):
        result[i] = run_test_for_single_y(parameter, i)
        if (i + 1) % (total / 5) == 0:
            print(f"{i + 1}/{total}")

    return result


def run_test_multiple(parameter: TestParameter, runs):
    result = numpy.zeros(parameter.test_data.n ** 2)
    start_time = time.time()
    for i in range(runs):
        print(f"Run #{i + 1}")
        result += run_test_for_all_y(parameter)
        print()
    print("{:.2f} s/run".format((time.time() - start_time) / runs))

    return result / runs
