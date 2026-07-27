import numpy as np
import matplotlib.pyplot as plt
from src.data_generation import generate_data
from src.model import LinearRegressionGD

# Генерация данных
X, y, X_train, X_test, y_train, y_test = generate_data(
    n_samples=200, weight=2, bias=5, noise=4
    )

# Создание модели и тренировка
model = LinearRegressionGD(n_features=1)
loss_history = model.train_model(
    X_train, y_train, learning_rate=0.0072, epochs=1000, verbose=True
    )

# Вывод итоговой MSE на тестовых данных
test_mse = model.score(X_test, y_test)
print(f'📊 MSE на тестовой выборке: {test_mse:.6f}')

# Итоговые параметры модели
w1, w0 = model.get_params()
print(f'📈 Выученные параметры: w1 = {w1[0]:.4f}, w0 = {w0:.4f}')

# График спада MSE
plt.figure(figsize=(10, 5))
plt.plot(loss_history, c='red', label='Линия MSE')
plt.title('Спад ошибки MSE')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()