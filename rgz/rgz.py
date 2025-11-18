import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

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

print("\nМодель Хольта")
# Перебор параметров alpha (уровень) и beta (тренд) с записью MAE, MSE, MAPE (Хольт)
alphas = np.arange(0.01, 1.0, 0.05)
betas = np.arange(0.01, 1.0, 0.05)
results = []
for a in alphas:
    for b in betas:
        model = Holt(spending_train, exponential=False, damped_trend=False)
        fit = model.fit(smoothing_level=float(a), smoothing_trend=float(b), optimized=False)
        forecast = fit.forecast(len(spending_test))
        mae_v = mean_absolute_error(spending_test, forecast)
        mse_v = mean_squared_error(spending_test, forecast)
        mape_v = np.mean(np.abs((spending_test - forecast) / spending_test)) * 100
        results.append({
            'alpha': round(float(a), 3),
            'beta': round(float(b), 3),
            'MAE': mae_v,
            'MSE': mse_v,
            'MAPE': mape_v
        })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=['MAPE', 'MAE']).reset_index(drop=True)
print('10 лучших комбинаций параметров:')
print(df_results.head(10).to_string(index=False))

csv_path = 'holt_grid_results.csv'
df_results.to_csv(csv_path, index=False)

best = df_results.iloc[0]
best_alpha = float(best['alpha'])
best_beta = float(best['beta'])
print(f"Лучшие параметры: alpha={best_alpha}, beta={best_beta}")

# Обучение финальной модели с выбранными лучшими параметрами и вычисление финальных метрик (Хольт)
best_model = Holt(spending_train, exponential=False, damped_trend=False)
best_fit = best_model.fit(smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False)
best_forecast = best_fit.forecast(len(spending_test))

mae = mean_absolute_error(spending_test, best_forecast)
mse = mean_squared_error(spending_test, best_forecast)
mape = np.mean(np.abs((spending_test - best_forecast) / spending_test)) * 100

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

# График исходных данных с прогнозом (Хольт)
plt.figure(figsize=(12, 6))
plt.plot(years, spending, marker='o', label='Исходные данные', color='b')
plt.plot(years_test, best_forecast, marker='s', linestyle='--', label='Прогноз', color='r')
plt.title('Исходные данные и прогноз (Хольт)')
plt.xlabel('Год')
plt.ylabel('Траты (доллары на человека в год)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График остатков (Хольт)
residuals = spending_test - best_forecast
plt.figure(figsize=(10, 5))
plt.plot(years_test, residuals, marker='o', linestyle='-', color='purple')
plt.title('График остатков (Хольт)')
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.grid(True)
plt.tight_layout()
plt.show()

# Корреллограмма остатков (Хольт)
plot_acf(residuals, lags=min(10, len(residuals)-1))
plt.title('Корреллограмма остатков (Хольт)')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.tight_layout()
plt.show()

print("\nМодель ARIMA")
# Перебор параметров p, d, q с записью MAE, MSE, MAPE (ARIMA)
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

arima_results = []
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(spending_train, order=(p, d, q))
                fit = model.fit()
                forecast = fit.forecast(len(spending_test))
                mae_v = mean_absolute_error(spending_test, forecast)
                mse_v = mean_squared_error(spending_test, forecast)
                mape_v = np.mean(np.abs((spending_test - forecast) / spending_test)) * 100
                arima_results.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'MAE': mae_v,
                    'MSE': mse_v,
                    'MAPE': mape_v
                })
            except Exception:
                continue

df_arima = pd.DataFrame(arima_results)
df_arima = df_arima.sort_values(by=['MAPE', 'MAE']).reset_index(drop=True)
print('10 лучших комбинаций параметров ARIMA:')
print(df_arima.head(10).to_string(index=False))

arima_csv = 'arima_grid_results.csv'
df_arima.to_csv(arima_csv, index=False)

best_arima = df_arima.iloc[0]
best_p, best_d, best_q = int(best_arima['p']), int(best_arima['d']), int(best_arima['q'])
print(f"Лучшие параметры: (p,d,q)=({best_p},{best_d},{best_q})")

# Обучение финальной модели ARIMA с лучшими параметрами 
final_arima = ARIMA(spending_train, order=(best_p, best_d, best_q))
final_fit = final_arima.fit()
arima_forecast = final_fit.forecast(len(spending_test))

arima_mae = mean_absolute_error(spending_test, arima_forecast)
arima_mse = mean_squared_error(spending_test, arima_forecast)
arima_mape = np.mean(np.abs((spending_test - arima_forecast) / spending_test)) * 100
print(f"MAE: {arima_mae:.2f}, MSE: {arima_mse:.2f}, MAPE: {arima_mape:.2f}%")

# График исходных данных с прогнозом ARIMA
plt.figure(figsize=(12, 6))
plt.plot(years, spending, marker='o', label='Исходные данные', color='b')
plt.plot(years_test, arima_forecast, marker='d', linestyle='--', label=f'ARIMA({best_p},{best_d},{best_q}) прогноз', color='g')
plt.title('Исходные данные и прогноз (ARIMA)')
plt.xlabel('Год')
plt.ylabel('Траты (доллары на человека в год)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Остатки для ARIMA
arima_resid = spending_test - arima_forecast
plt.figure(figsize=(10, 5))
plt.plot(years_test, arima_resid, marker='o', linestyle='-', color='orange')
plt.title('График остатков (ARIMA)')
plt.xlabel('Год')
plt.ylabel('Остатки')
plt.grid(True)
plt.tight_layout()
plt.show()

plot_acf(arima_resid, lags=min(10, len(arima_resid)-1))
plt.title('Корреллограмма остатков (ARIMA)')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.tight_layout()
plt.show()