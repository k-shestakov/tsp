import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_param = 2.0  # интенсивность экспоненциального распределения
num_trajectories = 10  # количество траекторий
t = np.linspace(0, 10, 500)  # временная сетка

# Корреляционная функция и нормированная корреляционная функция
t0 = 1.0  # фиксированное время

def R(t1, t2, lmbda=lambda_param):
    return lmbda / (lmbda + t1 + t2)

def rho(t1, t2, lmbda=lambda_param):
    return R(t1, t2, lmbda) / np.sqrt(R(t1, t1, lmbda) * R(t2, t2, lmbda))

# лаги
R_vals = [R(t0, t0 + d) for d in t]
rho_vals = [rho(t0, t0 + d) for d in t]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, R_vals, label=f"R(t, t+τ), t={t0}")
plt.xlabel("τ")
plt.ylabel("R")
plt.title("Корреляционная функция")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(t, rho_vals, label=f"ρ(t, t+τ), t={t0}")
plt.xlabel("τ")
plt.ylabel("ρ")
plt.title("Нормированная корреляционная функция")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Генерация траекторий и сохранение для корреляции
Y_all = []
for _ in range(num_trajectories):
	X = np.random.exponential(1/lambda_param)
	Y = np.exp(-X * t)
	Y_all.append(Y)
	plt.plot(t, Y, alpha=0.7)

plt.title(r"Семейство траекторий $Y(t) = e^{-Xt}$, $X \sim Exp(\lambda)$")
plt.xlabel('t')
plt.ylabel('Y(t)')
plt.grid(True)

# Математическое ожидание, дисперсия, среднеквадратическое отклонение
Ey = lambda_param / (lambda_param + t)
Ey2 = lambda_param / (lambda_param + 2 * t)
VarY = Ey2 - Ey**2
StdY = np.sqrt(VarY)

plt.figure()
plt.plot(t, Ey, label='Математическое ожидание E[Y(t)]')
plt.plot(t, VarY, label='Дисперсия Var[Y(t)]')
plt.plot(t, StdY, label='Среднеквадратичное отклонение Std[Y(t)]')
plt.xlabel('t')
plt.title('Моменты процесса Y(t)')
plt.legend()
plt.grid(True)
plt.show()