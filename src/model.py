import numpy as np

class LinearRegressionGD:

    def __init__(self, n_features=1):
        self.weights = np.zeros((n_features, 1)) # инициализация весов с подходящей размерностью(f, 1)
        self.bias = 0.0
        self.loss_history = []

    def predict(self, X):
        y_pred = X @ self.weights + self.bias # находим предсказание соблюдая размерности(n, 1)
        return y_pred

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_pred - y_true)**2)

    def train_model(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        self.loss_history = []
        n = X_train.shape[0]

        for epoch in range(epochs):
            y_pred = self.predict(X_train)

            # Вычисляем насколько сильно отличаются предсказания нашей модели от реальности
            loss = self.mean_squared_error(y_train, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y_train # вектор ошибок(n, 1)

            # Считаем градиенты для весов
            grad_weights = 2/n * (X_train.T @ error) # размерность(f, 1)
            grad_bias = 2/n * np.sum(error)

            # Вычисляем новые значения весов
            self.weights -= learning_rate*grad_weights
            self.bias -= learning_rate*grad_bias

        return self.loss_history
