import numpy as np
import matplotlib.pyplot as plt

lambda_param = 2.0
num_trajectories = 10
t = np.linspace(0, 10, 500)

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
def Ey_analytic(t, lam):
    return lam / (lam + t)

def VarY_analytic(t, lam):
    return lam / (lam + 2*t) - (lam / (lam + t))**2

def StdY_analytic(t, lam):
    return np.sqrt(VarY_analytic(t, lam))

Ey_np = Ey_analytic(t, lambda_param)
VarY_np = VarY_analytic(t, lambda_param)
StdY_np = StdY_analytic(t, lambda_param)

plt.figure(figsize=(10,6))
plt.plot(t, Ey_np, '--', label='E[Y(t)]', color='blue', alpha=0.6)
plt.plot(t, Ey_np - StdY_np, '--', label='E[Y(t)] - σ[Y(t)]', color='red', alpha=0.6)
plt.plot(t, Ey_np + StdY_np, '--', label='E[Y(t)] + σ[Y(t)]', color='green', alpha=0.6)
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

taus = t
R_mean = []
rho_mean = []
for tau in taus:
    vals_R = []
    vals_rho = []
    for t1 in t:
        t2 = t1 + tau
        if t2 <= t[-1]:
            vals_R.append(R(t1, t2))
            vals_rho.append(rho(t1, t2))
    if vals_R:
        R_mean.append(np.mean(vals_R))
        rho_mean.append(np.mean(vals_rho))
    else:
        R_mean.append(np.nan)
        rho_mean.append(np.nan)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(t, R_mean, label=f"R(t, t+τ), t={t0}")
plt.xlabel("τ")
plt.ylabel("R")
plt.title("Корреляционная функция")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)