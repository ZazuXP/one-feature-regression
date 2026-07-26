import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Создаём данные для обучения и разделяем их
def generate_data(n_samples=100, weight=2, bias=5, noise=4, seed=42):
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 20, n_samples)[:, np.newaxis]
    y_ideal = weight*X + bias
    y = y_ideal + rng.normal(0, noise, X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test

# Строим график для проверки данных
def plot_data(X, y):
    plt.figure(figsize=(8, 5))

    plt.scatter(X, y, alpha=0.7, label='Data with noise')
    plt.plot(X, 2*X + 5, ls='--', c='red', label='True line')
    plt.title('Generated data', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Target', fontsize=14)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Проверка генерации
if __name__ == '__main__':
    X, y, X_train, X_test, y_train, y_test = generate_data()
    print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    plot_data(X, y)