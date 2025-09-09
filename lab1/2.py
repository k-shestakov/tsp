# https://tylervigen.com/spurious/correlation/14685_associates-degrees-awarded-in-philosophy-and-religious-studies_correlates-with_cheddar-cheese-consumption

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

# Построение графиков для двух динамических рядов в одной системе координат
years = np.arange(1, len(array_1) + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Первый ряд на левой оси Y
ax1.plot(years, array_1, marker='o', color='tab:blue', label=array_1_name)
ax1.set_xlabel('Порядковый номер года')
ax1.set_ylabel(array_1_name, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Второй ряд на правой оси Y (та же ось X)
ax2 = ax1.twinx()
ax2.plot(years, array_2, marker='s', color='tab:orange', label=array_2_name)
ax2.set_ylabel(array_2_name, color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.suptitle('Динамические ряды (общая ось X, отдельные оси Y)')
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Показать график 
plt.show()

# Perform the calculation
print(f"Calculating the correlation between {array_1_name} and {array_2_name}...")
correlation, r_squared, p_value = calculate_correlation(array_1, array_2)

# Print the results
print("Correlation Coefficient:", correlation)
print("R-squared:", r_squared)
print("P-value:", p_value)