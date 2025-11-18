import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from statsmodels.tsa.holtwinters import Holt

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

# Grid search for two-parameter Holt (additive trend) smoothing
# We iterate over alpha (level) and beta (trend) values and record MAE, MSE, MAPE.
alphas = np.arange(0.01, 1.0, 0.05)
betas = np.arange(0.01, 1.0, 0.05)
results = []
for a in alphas:
    for b in betas:
        try:
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
        except Exception:
            # Some parameter combos may fail; skip them
            continue

# Build a comparative table and save it
df_results = pd.DataFrame(results)
if df_results.empty:
    raise RuntimeError('Grid search produced no valid fits. Try a different grid or enable optimization.')

df_results = df_results.sort_values(by=['MAPE', 'MAE']).reset_index(drop=True)
print('Top 10 parameter combinations by MAPE:')
print(df_results.head(10).to_string(index=False))

csv_path = 'holt_grid_results.csv'
df_results.to_csv(csv_path, index=False)
print(f"Saved grid search results to {csv_path}")

# Select best parameters by MAPE (primary) and MAE (secondary)
best = df_results.iloc[0]
best_alpha = float(best['alpha'])
best_beta = float(best['beta'])
print(f"Best params by MAPE: alpha={best_alpha}, beta={best_beta}")

# Fit final model with chosen best parameters and compute final metrics
best_model = Holt(spending_train, exponential=False, damped_trend=False)
best_fit = best_model.fit(smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False)
best_forecast = best_fit.forecast(len(spending_test))

mae = mean_absolute_error(spending_test, best_forecast)
mse = mean_squared_error(spending_test, best_forecast)
mape = np.mean(np.abs((spending_test - best_forecast) / spending_test)) * 100

print(f"Final model metrics with best params -> MAE: {mae:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

# График исходных данных с прогнозом
plt.figure(figsize=(12, 6))
plt.plot(years, spending, marker='o', label='Исходные данные', color='b')
plt.plot(years_test, best_forecast, marker='s', linestyle='--', label='Прогноз', color='r')
plt.title('Исходные данные и прогноз')
plt.xlabel('Год')
plt.ylabel('Траты (доллары на человека в год)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# График остатков 
residuals = spending_test - best_forecast
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

