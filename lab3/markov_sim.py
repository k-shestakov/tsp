# markov_sim.py
import numpy as np
import pandas as pd
import json

def simulate_markov(P, v0, m):
    v0 = np.array(v0, dtype=float).reshape(1,4)
    states = [v0.flatten()]
    v = v0.copy()
    for k in range(1, m+1):
        v = v @ P
        states.append(v.flatten())
    df = pd.DataFrame(states, columns=['s1','s2','s3','s4'])
    df.index.name = 'Цикл'
    return df

DEFAULT_P = [
    [0.85, 0.10, 0.03, 0.02],
    [0.60, 0.30, 0.05, 0.05],
    [0.50, 0.20, 0.25, 0.05],
    [0.80, 0.05, 0.05, 0.10],
]
DEFAULT_V0 = [1.0, 0.0, 0.0, 0.0]
DEFAULT_M = 10

if __name__ == "__main__":
    P = np.array(DEFAULT_P, dtype=float)
    v0 = np.array(DEFAULT_V0, dtype=float).reshape(-1)
    df = simulate_markov(P, v0, DEFAULT_M)
    print(df.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nВектор состояния после выполнения {} циклов: {}".format(DEFAULT_M, df.loc[DEFAULT_M].to_numpy().round(6)))
    print("Сумма вероятностей состояний: {}".format(df.loc[DEFAULT_M].to_numpy().sum().round(6)))
