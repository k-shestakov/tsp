import numpy as np
import matplotlib.pyplot as plt

lambda_param = 2.0  # интенсивность экспоненциального распределения
num_trajectories = 10  # количество траекторий
t = np.linspace(0, 10, 500)  # временная сетка

# Генерация траекторий и сохранение для корреляции
Y_all = []
for _ in range(num_trajectories):
    X = np.random.exponential(1/lambda_param)
    Y = np.exp(-X * t)
    Y_all.append(Y)
    plt.plot(t, Y, alpha=0.7)

Y_all_np = np.array(Y_all)

Ey_np = np.mean(Y_all_np, axis=0)
VarY_np = np.var(Y_all_np, axis=0)
StdY_np = np.std(Y_all_np, axis=0)

plt.title(r"Семейство траекторий $Y(t) = e^{-Xt}$, $X \sim Exp(\lambda)$")
plt.xlabel('t')
plt.ylabel('Y(t)')
plt.grid(True)

plt.figure(figsize=(10,6))
plt.plot(t, Ey_np, '--', label='Математическое ожидание: E[Y(t)]', color='blue', alpha=0.6)
plt.plot(t, VarY_np, '--', label='Дисперсия: Var[Y(t)]', color='red', alpha=0.6)
plt.plot(t, StdY_np, '--', label='Среднеквадратическое отклонение: Std[Y(t)]', color='green', alpha=0.6)
plt.xlabel('t')
plt.title('Моменты процесса Y(t)')
plt.legend()
plt.grid(True)
plt.show()

# Корреляционная функция и нормированная корреляционная функция
t0 = 1.0  

def R(t1, t2, lmbda=lambda_param):
    return lmbda / (lmbda + t1 + t2)

def rho(t1, t2, lmbda=lambda_param):
    return R(t1, t2, lmbda) / np.sqrt(R(t1, t1, lmbda) * R(t2, t2, lmbda))

# Лаги
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

is_stationary = np.allclose(Ey_np, Ey_np[0]) and np.allclose(VarY_np, VarY_np[0])
if is_stationary:
	stationary_text = "Процесс стационарен: его математическое ожидание и дисперсия не зависят от времени."
else:
	stationary_text = "Процесс нестационарен: его математическое ожидание и/или дисперсия зависят от времени."

ergodic_text = "Процесс эргодичен по среднему: среднее по времени совпадает с математическим ожиданием."

print("\nАнализ процесса Y(t) = exp(-X t), X ~ Exp(λ):")
print(stationary_text)
print(ergodic_text)