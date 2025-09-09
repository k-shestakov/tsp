
# Эти модули упрощают выполнение вычислений
import numpy as np
from scipy import stats

# Определим функцию, которую можно вызвать для возврата расчетов корреляции
def calculate_correlation(array1, array2):
    # Вычислить коэффициент корреляции Пирсона и p-значение
    correlation, p_value = stats.pearsonr(array1, array2)
    # Вычислить R-квадрат как квадрат коэффициента корреляции
    r_squared = correlation**2
    return correlation, r_squared, p_value


# Это массивы для переменных, показанных на этой странице, но вы можете изменить их на любые два набора чисел
import matplotlib.pyplot as plt

array_1 = np.array([283,308,326,435,697,814,1384,1357,1417,1625,1616,])
array_2 = np.array([9.59021,9.59326,9.64526,9.85696,10.176,10.4024,11.0865,11.2123,11.1567,11.1287,11.4113,])
array_1_name = "Количество выданных степеней по философии и религиоведению"
array_2_name = "Потребление сыра чеддер"



# Выполнить расчет
print(f"Вычисление корреляции между '{array_1_name}' и '{array_2_name}'...")
correlation, r_squared, p_value = calculate_correlation(array_1, array_2)

# Вывести результаты
print("Коэффициент корреляции:", correlation)
print("R-квадрат:", r_squared)
print("P-значение:", p_value)


# Расчет коэффициентов линейного тренда для каждого ряда по годам
years = np.arange(2011, 2022)
trend1 = np.polyfit(years, array_1, 1)  # [наклон, свободный член]
trend2 = np.polyfit(years, array_2, 1)

print(f"\nКоэффициенты линейного тренда для {array_1_name} (по годам):")
print(f"  Уравнение: y = {trend1[0]:.4f} * x + {trend1[1]:.4f}")
print(f"  Наклон: {trend1[0]:.4f}")
print(f"  Свободный член: {trend1[1]:.4f}")

print(f"\nКоэффициенты линейного тренда для {array_2_name} (по годам):")
print(f"  Уравнение: y = {trend2[0]:.4f} * x + {trend2[1]:.4f}")
print(f"  Наклон: {trend2[0]:.4f}")
print(f"  Свободный член: {trend2[1]:.4f}")


# Расчет значений по тренду для каждого ряда
trend_values_1 = trend1[0] * years + trend1[1]
trend_values_2 = trend2[0] * years + trend2[1]

print(f"\nЗначения по тренду для {array_1_name}:")
for year, value in zip(years, trend_values_1):
    print(f"  {year}: {value:.2f}")

print(f"\nЗначения по тренду для {array_2_name}:")
for year, value in zip(years, trend_values_2):
    print(f"  {year}: {value:.4f}")

# Остатки (разности между фактическими и трендовыми значениями)
residuals_1 = array_1 - trend_values_1
residuals_2 = array_2 - trend_values_2

print(f"\nОстатки для {array_1_name}:")
for year, value in zip(years, residuals_1):
    print(f"  {year}: {value:.2f}")

print(f"\nОстатки для {array_2_name}:")
for year, value in zip(years, residuals_2):
    print(f"  {year}: {value:.4f}")

# График остатков с двумя осями Y
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
ax1.set_xlabel('Год')
ax1.set_ylabel(f'Остатки {array_1_name}', color=color1)
ax1.plot(years, residuals_1, color=color1, marker='o', label=f'Остатки {array_1_name}')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel(f'Остатки {array_2_name}', color=color2)
ax2.plot(years, residuals_2, color=color2, marker='s', label=f'Остатки {array_2_name}')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.title('Остатки (две оси Y)')
fig.tight_layout()
plt.grid(True)
plt.show()

# Корреляция между остатками
correlation_resid, r2_resid, p_resid = calculate_correlation(residuals_1, residuals_2)
print(f"\nКорреляция между остатками:")
print("  Коэффициент корреляции:", correlation_resid)
print("  R-квадрат:", r2_resid)
print("  P-значение:", p_resid)

# Построение графика с двумя осями Y по годам
fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
ax1.set_xlabel('Год')
ax1.set_ylabel(array_1_name, color=color1)
ax1.plot(years, array_1, color=color1, marker='o', label=array_1_name)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel(array_2_name, color=color2)
ax2.plot(years, array_2, color=color2, marker='s', label=array_2_name)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Динамические ряды (две оси Y, по годам)')
fig.tight_layout()
plt.grid(True)
plt.show()