import matplotlib.pyplot as plt
import numpy as np

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

# Корреллограмма (автокорреляция)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(spending, lags=20)
plt.title('График автокорреляционной функции расходов на свежие фрукты')
plt.xlabel('Лаг')
plt.ylabel('Автокорреляция')
plt.tight_layout()
plt.show()

# График частной автокорреляционной функции (PACF)
plot_pacf(spending, lags=10, method='ywm')
plt.title('График частной автокорреляционной функции расходов на свежие фрукты')
plt.xlabel('Лаг')
plt.ylabel('Частная автокорреляция')
plt.tight_layout()
plt.show()
