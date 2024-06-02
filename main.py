import numpy as np
import matplotlib.pyplot as plt

def generate_target_function():
    points = np.random.uniform(-1, 1, (2, 2))
    a = points[1, 1] - points[0, 1]
    b = points[0, 0] - points[1, 0]
    c = points[1, 0] * points[0, 1] - points[0, 0] * points[1, 1]
    return a, b, c

def target_function(a, b, c, x):
    return np.sign(a * x[:, 0] + b * x[:, 1] + c)

def generate_data(N, a, b, c):
    X = np.random.uniform(-1, 1, (N, 2))
    y = target_function(a, b, c, X)
    return X, y

def perceptron_learning_algorithm(X, y):
    w = np.zeros(X.shape[1])
    b = 0
    iterations = 0
    while True:
        predictions = np.sign(np.dot(X, w) + b)
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X[idx]
        b += y[idx]
        iterations += 1
    return w, b, iterations

def calculate_disagreement(w, w_0, a, b, c, N=10000):
    X_test = np.random.uniform(-1, 1, (N, 2))
    y_test = target_function(a, b, c, X_test)
    y_pred = np.sign(np.dot(X_test, w) + w_0)
    disagreement = np.mean(y_test != y_pred)
    return disagreement, X_test, y_test

def experiment(num_runs, num_points):
    iterations_list = []
    disagreement_list = []
    for _ in range(num_runs):
        a, b, c = generate_target_function()
        X, y = generate_data(num_points, a, b, c)
        w, w_0, iterations = perceptron_learning_algorithm(X, y)
        disagreement, X_test, y_test = calculate_disagreement(w, w_0, a, b, c)
        iterations_list.append(iterations)
        disagreement_list.append(disagreement)
    return np.mean(iterations_list), np.mean(disagreement_list)

def plot_results(X, y, w, b, a, b_line, c_line):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    x_vals = np.linspace(-1, 1, 100)
    plt.plot(x_vals, -(a / b_line) * x_vals - c_line / b_line, 'k-', label='Target Function')
    plt.plot(x_vals, -(w[0] / w[1]) * x_vals - b / w[1], 'g-', label='Learned Function')
    plt.legend()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

# num_points = 10
# num_runs = 1000

# mean_iterations, mean_disagreement = experiment(num_runs, num_points)
# print(f"Média de iterações até a convergência: {mean_iterations}")
# print(f"Média de divergência entre f e g: {mean_disagreement}")

# a, b, c = generate_target_function()
# X, y = generate_data(num_points, a, b, c)
# w, w_0, iterations = perceptron_learning_algorithm(X, y)

# disagreement, X_test, y_test = calculate_disagreement(w, w_0, a, b, c)

# plot_results(X_test[:100], y_test[:100], w, w_0, a, b, c)

def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X] 
    w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 
    return w

def calculate_error(X, y, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = np.sign(X_b.dot(w))
    return np.mean(predictions != y)

def pla(X, y, w):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    iterations = 0
    while True:
        predictions = np.sign(X_b.dot(w))
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X_b[idx]
        iterations += 1
    return iterations

def experiment_rg(num_runs, num_points_train, num_points_test):
    ein_list = []
    eout_list = []
    for _ in range(num_runs):
        a, b, c = generate_target_function()
        X_train, y_train = generate_data(num_points_train, a, b, c)
        w = linear_regression(X_train, y_train)
        ein = calculate_error(X_train, y_train, w)
        ein_list.append(ein)
        X_test, y_test = generate_data(num_points_test, a, b, c)
        eout = calculate_error(X_test, y_test, w)
        eout_list.append(eout)
    return np.mean(ein_list), np.std(ein_list), np.mean(eout_list), np.std(eout_list)

def experiment_rg_pla(num_runs, num_points_train):
    iterations_list = []
    for _ in range(num_runs):
        a, b, c = generate_target_function()
        X_train, y_train = generate_data(num_points_train, a, b, c)
        w = linear_regression(X_train, y_train)
        iterations = pla(X_train, y_train, w)
        iterations_list.append(iterations)
    return np.mean(iterations_list), np.std(iterations_list)

# num_runs = 1000
# num_points_test = 1000
# num_points_train = 10

# mean_ein, std_ein, mean_eout, std_eout = experiment_rg(num_runs, num_points_train, num_points_test)
# print(f"Média de E_in: {mean_ein}")
# print(f"Desvio padrão de E_in: {std_ein}")
# print(f"Média de E_out: {mean_eout}")
# print(f"Desvio padrão de E_out: {std_eout}")

# mean_iterations, std_iterations = experiment_rg_pla(num_runs, num_points_train)
# print(f"Média de iterações até a convergência do PLA: {mean_iterations}")
# print(f"Desvio padrão de iterações: {std_iterations}")

def generate_noisy_data(N, a, b, c, noise_ratio=0.0):
    X = np.random.uniform(-1, 1, (N, 2))
    y = target_function(a, b, c, X)
    if noise_ratio > 0:
        num_noisy_points = int(N * noise_ratio)
        noisy_indices = np.random.choice(N, num_noisy_points, replace=False)
        y[noisy_indices] = -y[noisy_indices]
    return X, y

def pocket_pla(X, y, w_init, max_iterations):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    w_pocket = w_init
    best_error = np.mean(np.sign(X_b.dot(w_init)) != y)
    w = w_init.copy()
    for _ in range(max_iterations):
        predictions = np.sign(X_b.dot(w))
        misclassified = np.where(predictions != y)[0]
        if len(misclassified) == 0:
            break
        idx = np.random.choice(misclassified)
        w += y[idx] * X_b[idx]
        current_error = np.mean(np.sign(X_b.dot(w)) != y)
        if current_error < best_error:
            best_error = current_error
            w_pocket = w.copy()
    return w_pocket, best_error

def plot_results(X, y, w, a, b, c, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    x_vals = np.linspace(-1, 1, 100)
    plt.plot(x_vals, -(a / b) * x_vals - c / b, 'k-', label='Target Function')
    plt.plot(x_vals, -(w[1] / w[2]) * x_vals - w[0] / w[2], 'g-', label='Pocket PLA Hypothesis')
    plt.title(title)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend()
    plt.show()

def experiment(num_runs, num_points_train, num_points_test, max_iterations, initialize_with_linear_regression):
    ein_list = []
    eout_list = []
    for _ in range(num_runs):
        a, b, c = generate_target_function()
        X_train, y_train = generate_noisy_data(num_points_train, a, b, c, noise_ratio=0.1)
        if initialize_with_linear_regression:
            w_init = linear_regression(X_train, y_train)
        else:
            w_init = np.zeros(X_train.shape[1] + 1)
        w_pocket, ein = pocket_pla(X_train, y_train, w_init, max_iterations)
        X_test, y_test = generate_noisy_data(num_points_test, a, b, c)
        eout = calculate_error(X_test, y_test, w_pocket)
        ein_list.append(ein)
        eout_list.append(eout)
        if _ == 0:
            plot_results(X_test, y_test, w_pocket, a, b, c, f'Test Data with Pocket PLA Hypothesis\n{max_iterations} Iterations, {"Linear Regression" if initialize_with_linear_regression else "Zero"} Initialization')
    return np.mean(ein_list), np.std(ein_list), np.mean(eout_list), np.std(eout_list)

num_runs = 1000
num_points_train = 100
num_points_test = 1000

cases = [
    {"description": "(a) Inicializando com 0, i = 10; N1 = 100; N2 = 1000.", "max_iterations": 10, "initialize_with_linear_regression": False},
    {"description": "(b) Inicializando com 0, i = 50; N1 = 100; N2 = 1000.", "max_iterations": 50, "initialize_with_linear_regression": False},
    {"description": "(c) Inicializando com Regressão Linear, i = 10; N1 = 100; N2 = 1000.", "max_iterations": 10, "initialize_with_linear_regression": True},
    {"description": "(d) Inicializando com Regressão Linear, i = 50; N1 = 100; N2 = 1000.", "max_iterations": 50, "initialize_with_linear_regression": True},
]

for case in cases:
    mean_ein, std_ein, mean_eout, std_eout = experiment(num_runs, num_points_train, num_points_test, case["max_iterations"], case["initialize_with_linear_regression"])
    print(f"{case['description']}")
    print(f"  Média de E_in: {mean_ein}")
    print(f"  Desvio padrão de E_in: {std_ein}")
    print(f"  Média de E_out: {mean_eout}")
    print(f"  Desvio padrão de E_out: {std_eout}\n")

def non_linear_target_function(x1, x2):
    return np.sign(x1**2 + x2**2 - 0.6)

def generate_fixed_noisy_data(N):
    X = np.random.uniform(-1, 1, (N, 2))
    y = non_linear_target_function(X[:, 0], X[:, 1])
    num_noisy_points = int(N * 0.1)
    noisy_indices = np.random.choice(N, num_noisy_points, replace=False)
    y[noisy_indices] = -y[noisy_indices]
    return X, y

def experiment(num_runs, num_points):
    ein_list = []
    for _ in range(num_runs):
        X, y = generate_fixed_noisy_data(num_points)
        w = linear_regression(X, y)
        ein = calculate_error(X, y, w)
        ein_list.append(ein)
    return np.mean(ein_list), np.std(ein_list)

# num_runs = 1000
# num_points = 1000

# mean_ein, std_ein = experiment(num_runs, num_points)
# print(f"Média de E_in: {mean_ein}")
# print(f"Desvio padrão de E_in: {std_ein}")

def transform_data(X):
    x1, x2 = X[:, 0], X[:, 1]
    X_transformed = np.c_[np.ones(X.shape[0]), x1, x2, x1 * x2, x1**2, x2**2]
    return X_transformed

def linear_regression_simple(X, y):
    w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return w

def experiment(num_runs, num_points):
    weights_list = []
    for _ in range(num_runs):
        X, y = generate_fixed_noisy_data(num_points, noise_ratio=0.1)
        X_transformed = transform_data(X)
        w = linear_regression_simple(X_transformed, y)
        weights_list.append(w)
    weights_mean = np.mean(weights_list, axis=0)
    return weights_mean

# num_runs = 1000
# num_points = 1000

# weights_mean = experiment(num_runs, num_points)
# print(f"Pesos médios após 1000 execuções: {weights_mean}")

# hypotheses = {
#     "a": np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5]),
#     "b": np.array([-1, -0.05, 0.08, 0.13, 1.5, 15]),
#     "c": np.array([-1, -0.05, 0.08, 0.13, 15, 1.5]),
#     "d": np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05]),
#     "e": np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15]),
# }

# for key, hypothesis in hypotheses.items():
#     distance = np.linalg.norm(weights_mean - hypothesis)
#     print(f"Distância da hipótese {key}: {distance}")

# closest_hypothesis = min(hypotheses, key=lambda k: np.linalg.norm(weights_mean - hypotheses[k]))
# print(f"Hipótese mais próxima: {closest_hypothesis}")

def calculate_eout(X, y, w):
    predictions = np.sign(X.dot(w))
    return np.mean(predictions != y)

def experiment(num_runs, num_points_train, num_points_test):
    eout_list = []
    for _ in range(num_runs):
        X_train, y_train = generate_fixed_noisy_data(num_points_train)
        X_train_transformed = transform_data(X_train)
        w = linear_regression_simple(X_train_transformed, y_train)        
        X_test, y_test = generate_fixed_noisy_data(num_points_test, noise_ratio=0.0)
        X_test_transformed = transform_data(X_test)
        eout = calculate_eout(X_test_transformed, y_test, w)
        eout_list.append(eout)
    return np.mean(eout_list), np.std(eout_list)

# num_runs = 1000
# num_points_train = 1000
# num_points_test = 1000

# mean_eout, std_eout = experiment(num_runs, num_points_train, num_points_test)
# print(f"Média de E_out: {mean_eout}")
# print(f"Desvio padrão de E_out: {std_eout}")