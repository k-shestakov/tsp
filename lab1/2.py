# These modules make it easier to perform the calculation
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# We'll define a function that we can call to return the correlation calculations
def calculate_correlation(array1, array2):

    # Calculate Pearson correlation coefficient and p-value
    correlation, p_value = stats.pearsonr(array1, array2)

    # Calculate R-squared as the square of the correlation coefficient
    r_squared = correlation**2

    return correlation, r_squared, p_value

# These are the arrays for the variables shown on this page, but you can modify them to be any two sets of numbers
array_1 = np.array([283,308,326,435,697,814,1384,1357,1417,1625,1616,])
array_2 = np.array([9.59021,9.59326,9.64526,9.85696,10.176,10.4024,11.0865,11.2123,11.1567,11.1287,11.4113,])
array_1_name = "Associates degrees awarded in Philosophy and religious studies"
array_2_name = "Cheddar cheese consumption"

# Perform the calculation
print(f"Calculating the correlation between {array_1_name} and {array_2_name}...")
correlation, r_squared, p_value = calculate_correlation(array_1, array_2)

# Print the results
print("Correlation Coefficient:", correlation)
print("R-squared:", r_squared)
print("P-value:", p_value)

# Расчёт коэффициентов тренда для каждого ряда
trend_array_1 = np.polyfit(range(len(array_1)), array_1, 1)
trend_array_2 = np.polyfit(range(len(array_2)), array_2, 1)

# Вывод коэффициентов тренда
print(f"Коэффициенты тренда для {array_1_name}: {trend_array_1}")
print(f"Коэффициенты тренда для {array_2_name}: {trend_array_2}")

# Нормализация данных для корректного отображения на графике
normalized_array_1 = (array_1 - np.min(array_1)) / (np.max(array_1) - np.min(array_1))
normalized_array_2 = (array_2 - np.min(array_2)) / (np.max(array_2) - np.min(array_2))

# Построение графика данных динамических рядов с нормализацией
plt.figure(figsize=(10, 6))
plt.plot(normalized_array_1, label=array_1_name, marker='o')
plt.plot(normalized_array_2, label=array_2_name, marker='o')
plt.title("Динамические ряды (нормализованные)")
plt.xlabel("Индекс")
plt.ylabel("Нормализованное значение")
plt.legend()
plt.grid()
plt.show()