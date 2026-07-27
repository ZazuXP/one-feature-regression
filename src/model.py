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

    def train_model(self, X_train, y_train, learning_rate=0.01, epochs=1000, verbose=True):
        self.loss_history = []
        n = X_train.shape[0]

        for epoch in range(epochs):
            y_pred = self.predict(X_train)

            # Вычисляем насколько сильно отличаются предсказания нашей модели от реальности
            loss = self.mean_squared_error(y_train, y_pred)
            self.loss_history.append(loss)

            if np.isnan(loss) or np.isinf(loss):
                print(f'⚠️ Ошибка улетела в бесконечность на эпохе {epoch}!')
                print(f'   Последнее значение loss: {loss}')
                print(f'   Попробуй уменьшить learning_rate (сейчас {learning_rate})')
                break

            error = y_pred - y_train # вектор ошибок(n, 1)

            # Считаем градиенты для весов
            grad_weights = 2/n * (X_train.T @ error) # размерность(f, 1)
            grad_bias = 2/n * np.sum(error)

            # Вычисляем новые значения весов
            self.weights -= learning_rate*grad_weights
            self.bias -= learning_rate*grad_bias

            # Вывод прогресса обучения
            if verbose and (epoch+1)%100==0:
                print(f'Эпоха {epoch + 1}/{epochs}, MSE: {loss:.6f}')

        # Проверки и итоговый результат
        if verbose:
            if self.loss_history:
                final_loss = self.loss_history[-1]
                if np.isnan(final_loss) or np.isinf(final_loss):
                    print(f'\n⚠️ Обучение прервано из-за расходимости!')
                    print(f'   Последнее значение loss: {final_loss}')
                else:
                    print(f'\n✅ Обучение завершено! Финальная MSE: {self.loss_history[-1]:.6f}')
                    print(f'   Вес (w1): {self.weights[0][0]:.4f}')
                    print(f'   Смещение (w0): {self.bias:.4f}')
            else:
                print('\n⚠️ Обучение не выполнено (нет данных)!')

        return self.loss_history

    # Оценка на тестовых данных
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return self.mean_squared_error(y_test, y_pred)

    # Параметры модели (вес и смещение)
    def get_params(self):
        return self.weights.flatten(), self.bias
