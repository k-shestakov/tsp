import numpy as np
import matplotlib.pyplot as plt

lambda_param = 2.0  # Интенсивность экспоненциального распределения
num_trajectories = 10  # Количество траекторий
t = np.linspace(0, 10, 500)  # Временная сетка

# Генерация траекторий 
Y_all = []
for _ in range(num_trajectories):
    X = np.random.exponential(1/lambda_param)
    Y = np.exp(-X * t)
    Y_all.append(Y)
    plt.plot(t, Y, alpha=0.7)

Y_all_np = np.array(Y_all)

plt.title(r"Семейство траекторий $Y(t) = e^{-Xt}$, $X \sim Exp(\lambda)$")
plt.xlabel('t')
plt.ylabel('Y(t)')
plt.grid(True)
plt.show()

print(f"Траектории процесса:\n{Y_all_np}\n")

# Моменты процесса
Ey_np = np.mean(Y_all_np, axis=0)
VarY_np = np.var(Y_all_np, axis=0)
StdY_np = np.std(Y_all_np, axis=0)

plt.figure(figsize=(10,6))
plt.plot(t, Ey_np, '--', label='Математическое ожидание', color='blue', alpha=0.6)
plt.plot(t, StdY_np, '--', label='Среднеквадратическое отклонение', color='green', alpha=0.6)
plt.xlabel('t')
plt.title('Моменты процесса Y(t)')
plt.legend()
plt.grid(True)
plt.show()

print("Моменты процесса Y(t):")
print(f"Математическое ожидание:\n{Ey_np}\n")
print(f"Дисперсия:\n{VarY_np}\n")
print(f"Среднеквадратическое отклонение:\n{StdY_np}\n")

# Корреляционная функция и нормированная корреляционная функция
t0 = 1.0  

def R(t1, t2):
    return lambda_param / (lambda_param + t1 + t2)

def rho(t1, t2):
    return np.sqrt((lambda_param + 2*t1) * (lambda_param + 2*t2)) / (lambda_param + t1 + t2)

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

print("Анализ процесса Y(t) = exp(-X t), X ~ Exp(λ):")
print("Процесс нестационарен: его математическое ожидание и дисперсия зависят от времени.")
print("Процесс эргодичен по среднему: среднее по времени совпадает с математическим ожиданием.")