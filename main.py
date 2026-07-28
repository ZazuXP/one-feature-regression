import matplotlib.pyplot as plt
from src.data_generation import generate_data
from src.model import LinearRegressionGD
from visualization import plot_loss_history, plot_predictions_vs_actuals, plot_residuals_histogram, plot_all_graphics

# Генерация данных
X, y, X_train, X_test, y_train, y_test = generate_data(
    n_samples=500, weight=2, bias=5, noise=4
    )

# Создание модели и тренировка
model = LinearRegressionGD(n_features=1)
loss_history = model.train_model(
    X_train, y_train, learning_rate=0.0071, epochs=1000, verbose=True
    )

# Вывод итоговой MSE на тестовых данных
test_mse = model.score(X_test, y_test)
print(f'📊 MSE на тестовой выборке: {test_mse:.6f}')

# Итоговые параметры модели
w1, w0 = model.get_params()
print(f'📈 Выученные параметры: w1 = {w1[0]:.4f}, w0 = {w0:.4f}')

# График спада MSE
plot_all_graphics(X, y, model, loss_history)