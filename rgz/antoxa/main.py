import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
FIG_DIR = "figures"
RES_DIR = "results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Загрузка временного ряда
df = pd.read_csv("apple_stock.csv", parse_dates=["Date"], dayfirst=False)
df = df[["Date", "Close"]].sort_values("Date").reset_index(drop=True)
print("Данные загружены. Диапазон дат:", df["Date"].min(), "-", df["Date"].max())
print("Всего наблюдений:", len(df))

# Агрегация в недельный ряд
series_daily = df.set_index("Date")["Close"].asfreq(None)  
series_weekly = series_daily.resample('W-FRI').last().dropna()  

series = series_weekly.copy()
series.name = "Close"

series.to_csv(os.path.join(RES_DIR, "series_weekly.csv"))

# График ряда
plt.figure(figsize=(14,5))
plt.plot(series, label='Weekly Close', color='tab:blue')
plt.title('Apple Weekly Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "series_weekly.png"))
plt.close()

print("График временного ряда сохранён:", os.path.join(FIG_DIR, "series_weekly.png"))

# STL decomposition график
from statsmodels.tsa.seasonal import STL
stl = STL(series, robust=True)
res_stl = stl.fit()
fig = res_stl.plot()
fig.set_size_inches(14, 8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "stl_decomposition.png"))
plt.close()
print("STL decomposition график сохранён:", os.path.join(FIG_DIR, "stl_decomposition.png"))

# Анализ выбросов с использованием метода межквартильного размаха на разностях
diff = series.diff().dropna()

Q1 = diff.quantile(0.25)
Q3 = diff.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_mask = (diff < lower) | (diff > upper)
outliers_idx = diff.index[outliers_mask]
outliers_values = series.loc[outliers_idx]

# Сохраняем список выбросов
outliers_df = pd.DataFrame({"Date": outliers_idx, "Close": series.loc[outliers_idx].values, "Diff": diff.loc[outliers_idx].values})
outliers_df.to_csv(os.path.join(RES_DIR, "outliers.csv"), index=False)

# Визуализация: исходный ряд с пометкой выбросов
plt.figure(figsize=(14,5))
plt.plot(series, label='Weekly Close', color='tab:blue')
plt.scatter(outliers_idx, series.loc[outliers_idx], color='red', label='Outliers', zorder=5)
plt.title('Detected Outliers (IQR on differences)')
plt.xlabel('Date'); plt.ylabel('Close Price'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "outliers_on_series.png"))
plt.close()

print(f"Найдено выбросов (IQR на разностях): {len(outliers_idx)}. Список сохранён в results/outliers.csv")

# Сохранение ряда без выбросов (замена медианой в окне ±3 недели)
series_no_outliers = series.copy()
for idx in outliers_idx:
    window = series.loc[idx - pd.Timedelta(days=21): idx + pd.Timedelta(days=21)]
    med = window.median()
    series_no_outliers.loc[idx] = med

series_no_outliers.to_csv(os.path.join(RES_DIR, "series_weekly_no_outliers.csv"))

# ACF/PACF исходного ряда
fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(series.dropna(), lags=60, ax=ax[0], title='ACF of Weekly Close')
plot_pacf(series.dropna(), lags=60, ax=ax[1], title='PACF of Weekly Close', method='ywm')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "acf_pacf_series.png"))
plt.close()

# ACF/PACF разностей (для выявления AR/MA после устранения тренда)
fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(series.diff().dropna(), lags=60, ax=ax[0], title='ACF of First Differences')
plot_pacf(series.diff().dropna(), lags=60, ax=ax[1], title='PACF of First Differences', method='ywm')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "acf_pacf_diff.png"))
plt.close()

print("ACF/PACF графики сохранены.")

# Тест Дики-Фуллера для проверки стационарности
def adf_test(series_in, title='ADF Test'):
    print(f"\n--- {title} ---")
    res = adfuller(series_in.dropna(), autolag='AIC')
    print(f"ADF Statistic: {res[0]:.4f}")
    print(f"p-value: {res[1]:.4f}")
    print("Used lags:", res[2])
    print("Number of observations used:", res[3])
    return res

adf_orig = adf_test(series, "ADF Test on Original Series")
adf_diff = adf_test(series.diff().dropna(), "ADF Test on First Differences")

# Разделение на train/test (80/20)
n = len(series)
train_size = int(n * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Метрики
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape_val = mape(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape_val}

# SARIMA
# Определяем сезонный период m
m = 52 

# Автоматический подбор SARIMA для начальной оценки параметров
print("\nAuto ARIMA (stepwise) для предварительного подбора параметров SARIMA...")
auto_model = pm.auto_arima(train,
                           seasonal=True, m=m,
                           start_p=0, start_q=0, max_p=5, max_q=5,
                           start_P=0, start_Q=0, max_P=2, max_Q=2,
                           d=None, D=None,
                           trace=True, error_action='ignore', suppress_warnings=True,
                           stepwise=True, information_criterion='aic')

print("Auto ARIMA summary:")
print(auto_model.summary())

p_auto, d_auto, q_auto = auto_model.order
P_auto, D_auto, Q_auto, m_auto = auto_model.seasonal_order
print("Auto-selected order:", (p_auto, d_auto, q_auto), "seasonal_order:", (P_auto, D_auto, Q_auto, m_auto))

# Сеточный перебор SARIMA 
import itertools
sarima_results = []
p_range = range(0, 4)
d_range = [0, 1]
q_range = range(0, 4)
P_range = range(0, 2)
D_range = [0, 1]
Q_range = range(0, 2)

# Ограничение: не перебираем все комбинации, если их слишком много; остановимся после N успешных моделей
MAX_MODELS = 60
count = 0

print("\nGrid search SARIMA (limited) — начинаем перебор...")
for (p, d, q) in itertools.product(p_range, d_range, q_range):
    for (P, D, Q) in itertools.product(P_range, D_range, Q_range):
        if count >= MAX_MODELS:
            break
        try:
            # Используем SARIMAX с enforce_stationarity=False для гибкости
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            # Прогноз на длину теста
            pred = res.get_forecast(steps=len(test))
            fcast = pred.predicted_mean
            metrics = evaluate_forecast(test, fcast)
            # Ljung-Box тест на остатках модели (на обучении)
            lb = acorr_ljungbox(res.resid.dropna(), lags=[10], return_df=True)
            lb_pvalue = float(lb["lb_pvalue"].iloc[0])
            sarima_results.append({
                "p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q,
                "AIC": res.aic, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"],
                "LB_pvalue": lb_pvalue
            })
            count += 1
            if count % 10 == 0:
                print(f"  Протестировано моделей: {count}")
        except Exception as e:
            continue
    if count >= MAX_MODELS:
        break

sarima_df = pd.DataFrame(sarima_results).sort_values("RMSE").reset_index(drop=True)
sarima_df.to_excel(os.path.join(RES_DIR, "sarima_grid_results.xlsx"), index=False)
print("Grid search SARIMA завершён. Результаты сохранены в results/sarima_grid_results.xlsx")
print("Топ-5 моделей по RMSE:")
print(sarima_df.head(5))

# Выберем лучшую модель по RMSE для дальнейного анализа
best_sarima = sarima_df.iloc[0]
best_order = (int(best_sarima.p), int(best_sarima.d), int(best_sarima.q))
best_seasonal = (int(best_sarima.P), int(best_sarima.D), int(best_sarima.Q), m)
print("Лучшая SARIMA по RMSE:", best_order, best_seasonal)

# Обучаем финальную модель SARIMA на train
model_sarima = SARIMAX(train, order=best_order, seasonal_order=best_seasonal,
                       enforce_stationarity=False, enforce_invertibility=False)
res_sarima = model_sarima.fit(disp=False)
pred_sarima = res_sarima.get_forecast(steps=len(test))
fcast_sarima = pred_sarima.predicted_mean
conf_sarima = pred_sarima.conf_int()

# Сохраняем прогноз и метрики
metrics_sarima = evaluate_forecast(test, fcast_sarima)
metrics_sarima["Model"] = f"SARIMA{best_order}{best_seasonal}"
print("SARIMA metrics:", metrics_sarima)

# График: исходный ряд, модельные значения и прогноз
fitted_sarima = res_sarima.fittedvalues
plt.figure(figsize=(14,6))
plt.plot(series, label='Observed', color='black')
plt.plot(fitted_sarima.index, fitted_sarima, label='Fitted (SARIMA)', color='tab:orange')
plt.plot(fcast_sarima.index, fcast_sarima, label='Forecast (SARIMA)', color='tab:green')
plt.fill_between(conf_sarima.index, conf_sarima.iloc[:,0], conf_sarima.iloc[:,1], color='green', alpha=0.2)
plt.axvline(test.index[0], color='gray', linestyle='--', label='Train/Test split')
plt.title('SARIMA: Fitted and Forecast')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sarima_fit_forecast.png"))
plt.close()

# Диагностика остатков
res_sarima.plot_diagnostics(figsize=(12,10))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sarima_diagnostics.png"))
plt.close()

# Сохраняем остатки
resid_sarima = res_sarima.resid.dropna()
resid_sarima.to_csv(os.path.join(RES_DIR, "sarima_residuals.csv"))

# Модель трехступенчатого экспоненциального сглаживания 
# Для недельного ряда годовая сезонность m=52
seasonal_type = 'add'

hw_model = ExponentialSmoothing(train, trend='add', seasonal=seasonal_type, seasonal_periods=m)
hw_res = hw_model.fit(optimized=True)
hw_fcast = hw_res.forecast(len(test))

metrics_hw = evaluate_forecast(test, hw_fcast)
metrics_hw["Model"] = f"HoltWinters_{seasonal_type}"
print("Holt-Winters metrics:", metrics_hw)

# График 
plt.figure(figsize=(14,6))
plt.plot(series, label='Observed', color='black')
plt.plot(hw_res.fittedvalues.index, hw_res.fittedvalues, label='Fitted (HW)', color='tab:orange')
plt.plot(hw_fcast.index, hw_fcast, label='Forecast (HW)', color='tab:green')
plt.axvline(test.index[0], color='gray', linestyle='--', label='Train/Test split')
plt.title('Holt-Winters: Fitted and Forecast')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hw_fit_forecast.png"))
plt.close()

# Остатки 
resid_hw = (train - hw_res.fittedvalues).dropna()
plt.figure(figsize=(12,4))
plot_acf(resid_hw, lags=60, ax=plt.gca(), title='ACF of HW residuals')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hw_resid_acf.png"))
plt.close()

resid_hw.to_csv(os.path.join(RES_DIR, "hw_residuals.csv"))

# Сравнительный анализ моделей
comparison = pd.DataFrame([metrics_sarima, metrics_hw])
cols = ["Model", "MAE", "MSE", "RMSE", "MAPE"]
comparison = comparison[cols]
comparison.to_excel(os.path.join(RES_DIR, "models_comparison.xlsx"), index=False)
print("\nСравнение моделей сохранено в results/models_comparison.xlsx")
print(comparison)

# Визуализация прогнозов обеих моделей на тестовой выборке
plt.figure(figsize=(14,6))
plt.plot(train.index, train, label='Train', color='black')
plt.plot(test.index, test, label='Test', color='gray')
plt.plot(fcast_sarima.index, fcast_sarima, label='SARIMA Forecast', color='tab:green')
plt.plot(hw_fcast.index, hw_fcast, label='HW Forecast', color='tab:purple')
plt.title('Model Forecasts Comparison')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "models_forecast_comparison.png"))
plt.close()

# Исследование чувствительности моделей
# Выберем d и D по результатам ADF и auto_arima
# Если auto_model предложил d_auto, D_auto - используем их, иначе используем d=1 если ADF показал нестационарность.
d_used = d_auto if d_auto is not None else (1 if adf_orig[1] > 0.05 else 0)
D_used = D_auto if D_auto is not None else (1 if adf_orig[1] > 0.05 else 0)

# Для сезонных orders используем best_seasonal найденный ранее (или auto)
seasonal_fixed = best_seasonal
P_fixed, D_fixed, Q_fixed, m_fixed = seasonal_fixed

print("\nSARIMA sensitivity: фиксируем сезонную часть:", seasonal_fixed, "и d,D:", d_used, D_used)

p_vals = range(0,5)
q_vals = range(0,5)
sarima_sens_results = []

for p in p_vals:
    for q in q_vals:
        try:
            model = SARIMAX(train, order=(p, d_used, q), seasonal_order=(P_fixed, D_fixed, Q_fixed, m_fixed),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            f = res.get_forecast(steps=len(test)).predicted_mean
            rmse_val = np.sqrt(mean_squared_error(test, f))
            sarima_sens_results.append({"p": p, "q": q, "RMSE": rmse_val, "AIC": res.aic})
        except Exception:
            sarima_sens_results.append({"p": p, "q": q, "RMSE": np.nan, "AIC": np.nan})

sens_df = pd.DataFrame(sarima_sens_results)
pivot_rmse = sens_df.pivot(index='p', columns='q', values='RMSE')

plt.figure(figsize=(8,6))
sns.heatmap(pivot_rmse, annot=True, fmt=".0f", cmap="viridis")
plt.title(f"SARIMA RMSE heatmap (seasonal fixed {seasonal_fixed}, d={d_used}, D={D_used})")
plt.xlabel("q"); plt.ylabel("p")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sarima_pq_rmse_heatmap.png"))
plt.close()

sens_df.to_excel(os.path.join(RES_DIR, "sarima_pq_sensitivity.xlsx"), index=False)
print("SARIMA p/q sensitivity results saved to results/sarima_pq_sensitivity.xlsx and heatmap saved.")

# Фиксируем p,q как в best_order и исследуем P,Q
p_fix, d_fix, q_fix = best_order
P_vals = range(0,3)
Q_vals = range(0,3)
seasonal_sens_results = []

print("\nSARIMA seasonal sensitivity (P,Q) for fixed p,q:", best_order)
for P in P_vals:
    for Q in Q_vals:
        try:
            model = SARIMAX(train, order=(p_fix, d_fix, q_fix), seasonal_order=(P, D_used, Q, m_fixed),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            f = res.get_forecast(steps=len(test)).predicted_mean
            rmse_val = np.sqrt(mean_squared_error(test, f))
            seasonal_sens_results.append({"P": P, "Q": Q, "RMSE": rmse_val, "AIC": res.aic})
        except Exception:
            seasonal_sens_results.append({"P": P, "Q": Q, "RMSE": np.nan, "AIC": np.nan})

seasonal_df = pd.DataFrame(seasonal_sens_results)
pivot_seasonal_rmse = seasonal_df.pivot(index='P', columns='Q', values='RMSE')

plt.figure(figsize=(6,5))
sns.heatmap(pivot_seasonal_rmse, annot=True, fmt=".0f", cmap="magma")
plt.title(f"SARIMA seasonal RMSE heatmap (p,q fixed {best_order}, d={d_fix}, D={D_used})")
plt.xlabel("Q"); plt.ylabel("P")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "sarima_PQ_rmse_heatmap.png"))
plt.close()

seasonal_df.to_excel(os.path.join(RES_DIR, "sarima_PQ_sensitivity.xlsx"), index=False)
print("SARIMA seasonal sensitivity results saved and heatmap created.")

# Исследуем влияние параметров сглаживания alpha, beta, gamma.
# Обоснование диапазонов: значения в (0.01, 0.99). Для экономии времени используем сетку:
alphas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9]
betas  = [0.0, 0.01, 0.05, 0.1, 0.2]
gammas = [0.0, 0.01, 0.05, 0.1, 0.2]

hw_sens_results = []
print("\nHolt-Winters sensitivity grid search (alpha, beta, gamma)...")
for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            try:
                hw = ExponentialSmoothing(train, trend='add', seasonal=seasonal_type, seasonal_periods=m)
                res = hw.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma, optimized=False)
                f = res.forecast(len(test))
                rmse_val = np.sqrt(mean_squared_error(test, f))
                hw_sens_results.append({"alpha": alpha, "beta": beta, "gamma": gamma, "RMSE": rmse_val})
            except Exception:
                hw_sens_results.append({"alpha": alpha, "beta": beta, "gamma": gamma, "RMSE": np.nan})

hw_sens_df = pd.DataFrame(hw_sens_results)
hw_sens_df.to_excel(os.path.join(RES_DIR, "hw_sensitivity.xlsx"), index=False)
print("Holt-Winters sensitivity results saved to results/hw_sensitivity.xlsx")

# Визуализация: для нескольких фиксированных gamma строим heatmap alpha vs beta
for gamma in sorted(hw_sens_df['gamma'].unique()):
    subset = hw_sens_df[hw_sens_df['gamma'] == gamma]
    pivot = subset.pivot(index='alpha', columns='beta', values='RMSE')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm")
    plt.title(f"HW RMSE heatmap (gamma={gamma})")
    plt.xlabel("beta"); plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"hw_alpha_beta_gamma_{gamma}.png"))
    plt.close()

print("Holt-Winters sensitivity heatmaps сохранены в папке figures.")

# Сохраняем ключевые результаты в один Excel-файл для удобства
with pd.ExcelWriter(os.path.join(RES_DIR, "full_results_summary.xlsx")) as writer:
    series.to_frame().to_excel(writer, sheet_name="series", index=True)
    outliers_df.to_excel(writer, sheet_name="outliers", index=False)
    sarima_df.to_excel(writer, sheet_name="sarima_grid", index=False)
    sens_df.to_excel(writer, sheet_name="sarima_pq_sens", index=False)
    seasonal_df.to_excel(writer, sheet_name="sarima_PQ_sens", index=False)
    hw_sens_df.to_excel(writer, sheet_name="hw_sens", index=False)
    comparison.to_excel(writer, sheet_name="models_comparison", index=False)

print("\nПолный свод результатов сохранён в results/full_results_summary.xlsx")