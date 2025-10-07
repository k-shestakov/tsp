# https://www.tylervigen.com/spurious/correlation/5904_cheddar-cheese-consumption_correlates-with_solar-power-generated-in-haiti

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Функция для расчета корреляции
def calculate_correlation(array1, array2):
    correlation = stats.pearsonr(array1, array2)
    return correlation[0]

# Динамические ряды
array_1 = np.array([9.59326,9.64526,9.85696,10.176,10.4024,11.0865,11.2123,11.1567,11.1287,11.4113,])
array_2 = np.array([0.0009,0.0009,0.0009,0.002,0.002,0.003,0.003,0.003,0.003,0.00366,])
array_1_name = "Потребление сыра Чеддер"
array_2_name = "Выработка солнечной энергии в Гаити"

# Выполнение расчета
correlation = calculate_correlation(array_1, array_2)

# Печать результатов
print(f"Коэффициент корреляции для {array_1_name} и {array_2_name}:", correlation)

# Построение графика динамических рядов 
fig, ax1 = plt.subplots()
x = np.arange(1, len(array_1) + 1)

color1 = 'tab:blue'
ax1.set_xlabel('Наблюдение')
ax1.set_ylabel(array_1_name, color=color1)
ax1.plot(x, array_1, color=color1, marker='o', label=array_1_name)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel(array_2_name, color=color2)
ax2.plot(x, array_2, color=color2, marker='s', label=array_2_name)
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title('Динамические ряды')
plt.show()

# Второй график: обе оси y начинаются с 0
fig2, ax1_2 = plt.subplots()
ax1_2.set_xlabel('Наблюдение')
ax1_2.set_ylabel(array_1_name, color=color1)
ax1_2.plot(x, array_1, color=color1, marker='o', label=array_1_name)
ax1_2.tick_params(axis='y', labelcolor=color1)
ax1_2.set_ylim(bottom=0)

ax2_2 = ax1_2.twinx()
ax2_2.set_ylabel(array_2_name, color=color2)
ax2_2.plot(x, array_2, color=color2, marker='s', label=array_2_name)
ax2_2.tick_params(axis='y', labelcolor=color2)
ax2_2.set_ylim(bottom=0)

fig2.tight_layout()
plt.title('Динамические ряды (оси Y от 0)')
plt.show()

# Расчет коэффициентов тренда (a и b) для уравнения y = ax + b
x = np.arange(1, len(array_1) + 1)

# Для array_1
coeffs1 = np.polyfit(x, array_1, 1)
a1, b1 = coeffs1[0], coeffs1[1]
print(f"\nКоэффициенты тренда для {array_1_name}: a = {a1:.4f}, b = {b1:.3f}")

# Для array_2
coeffs2 = np.polyfit(x, array_2, 1)
a2, b2 = coeffs2[0], coeffs2[1]
print(f"\nКоэффициенты тренда для {array_2_name}: a = {a2:.4f}, b = {b2:.4f}")

# Расчет значений по тренду для каждого ряда
trend_1 = a1 * x + b1
trend_2 = a2 * x + b2
print(f"\nЗначения по тренду для {array_1_name}: {np.round(trend_1, 4)}")
print(f"\nЗначения по тренду для {array_2_name}: {np.round(trend_2, 5)}")

# Расчет остатков
residuals_1 = array_1 - trend_1
residuals_2 = array_2 - trend_2
print(f"\nОстатки для {array_1_name}: {np.round(residuals_1, 4)}")
print(f"\nОстатки для {array_2_name}: {np.round(residuals_2, 5)}")

# Корреляция между остатками
resid_corr = calculate_correlation(residuals_1, residuals_2)
print(f"\nКорреляция между остатками: {resid_corr}")