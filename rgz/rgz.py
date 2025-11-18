import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Данные временного ряда
years = np.array([
    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021
])
spending = np.array([
    163, 160, 178, 171, 187, 182, 195, 202, 222, 220, 232, 247, 261, 270, 274, 284, 288, 314, 318, 322, 349, 378
])

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(years, spending, marker='o', linestyle='-', color='b')
plt.title('Годовые расходы домохозяйств США на свежие фрукты')
plt.xlabel('Год')
plt.ylabel('Траты (доллары на человека в год)')
plt.grid(True)
plt.tight_layout()
plt.show()

# График автокорреляционной функции
plot_acf(spending, lags=20)
plt.title('График автокорреляционной функции расходов на свежие фрукты')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.tight_layout()
plt.show()

# График частной автокорреляционной функции
plot_pacf(spending, lags=10, method='ywm')
plt.title('График частной автокорреляционной функции расходов на свежие фрукты')
plt.xlabel('Лаг')
plt.ylabel('Частная автокорреляция')
plt.tight_layout()
plt.show()

# Поиск выбросов методом межквартильного размаха
Q1 = np.percentile(spending, 25)
Q3 = np.percentile(spending, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_indices = [i for i, value in enumerate(spending) if value < lower_bound or value > upper_bound]
print("Индексы выбросов в исходном ряде:")
print(outlier_indices)

# Разделение на обучающую и тестовую выборки (80/20)
split_idx = int(len(spending) * 0.8)
years_train = years[:split_idx]
years_test = years[split_idx:]
spending_train = spending[:split_idx]
spending_test = spending[split_idx:]

# Обучение модели на обучающей выборке
holt_model = ExponentialSmoothing(spending_train, trend='add', seasonal=None)
holt_fit = holt_model.fit()

# Прогноз на период тестовой выборки
holt_forecast = holt_fit.forecast(len(spending_test))

mae = mean_absolute_error(spending_test, holt_forecast)
mse = mean_squared_error(spending_test, holt_forecast)
mape = np.mean(np.abs((spending_test - holt_forecast) / spending_test)) * 100

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}%")

# График исходных данных с прогнозом
plt.figure(figsize=(12, 6))
plt.plot(years, spending, marker='o', label='Исходные данные', color='b')
plt.plot(years_test, holt_forecast, marker='s', linestyle='--', label='Прогноз', color='r')
plt.title('Исходные данные и прогноз')
plt.xlabel('Год')
plt.ylabel('Траты (доллары на человека в год)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График остатков 
residuals = spending_test - holt_forecast
plt.figure(figsize=(10, 5))
plt.plot(years_test, residuals, marker='o', linestyle='-', color='purple')
plt.title('График остатков')
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.grid(True)
plt.tight_layout()
plt.show()

# Корреллограмма остатков
plot_acf(residuals, lags=min(10, len(residuals)-1))
plt.title('Корреллограмма остатков')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.tight_layout()
plt.show()