import matplotlib.pyplot as plt
import numpy as np

# График спада MSE
def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, c='red', linewidth=1.5, label='Линия MSE')
    plt.title('Спад ошибки MSE', fontsize=16, fontweight='bold')
    plt.xlabel('Эпоха', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# График отклонения предсказаний от реальности (Scatter plot)
def plot_predictions_vs_actuals(X, y, model):
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, c='blue', alpha=0.5, label='Предсказания')

    # Идеальная линия (y=x)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], ls='--', c='red', linewidth=1.5, label='Линия y=x')

    plt.title('Отклонения предсказаний от реальности', fontsize=16, fontweight='bold')
    plt.xlabel('Действительные значения', fontsize=14)
    plt.ylabel('Предсказанные значения', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Гистограмма остатков
def plot_residuals_histogram(X, y, model):
    y_pred = model.predict(X)
    residuals = y.flatten() - y_pred.flatten()
    std = np.std(residuals)

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=25, color='blue', edgecolor='black', linewidth=0.5, alpha=0.7,)
    plt.title(f'Гистограмма остатков\nσ = {std:.3f}', fontsize=16, fontweight='bold')
    plt.xlabel('Остатки', fontsize=14)
    plt.ylabel('Частота', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all_graphics(X, y, model, loss_history):

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(loss_history, c='red', linewidth=1.5, label='Линия MSE')
    plt.title('Спад ошибки MSE', fontsize=16, fontweight='bold')
    plt.xlabel('Эпоха', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    y_pred = model.predict(X)
    plt.scatter(y, y_pred, c='blue', alpha=0.5, label='Предсказания')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], ls='--', c='red', linewidth=1.5, label='Линия y=x')
    plt.title('Отклонения предсказаний от реальности', fontsize=16, fontweight='bold')
    plt.xlabel('Действительные значения', fontsize=14)
    plt.ylabel('Предсказанные значения', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    residuals = y.flatten() - y_pred.flatten()
    std = np.std(residuals)
    plt.hist(residuals, bins=25, color='blue', edgecolor='black', linewidth=0.5, alpha=0.7,)
    plt.title(f'Гистограмма остатков\nσ = {std:.3f}', fontsize=16, fontweight='bold')
    plt.xlabel('Остатки', fontsize=14)
    plt.ylabel('Частота', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()