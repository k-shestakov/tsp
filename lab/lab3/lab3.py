# markov_sim.py
import numpy as np
import pandas as pd
import json

def validate_transition_matrix(P, tolerance=1e-6):
    """
    Проверка корректности матрицы переходов
    """
    P = np.array(P, dtype=float)
    
    # Проверка 1: Матрица должна быть квадратной
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Матрица переходов должна быть квадратной. Получен размер: {P.shape}")
    
    # Проверка 2: Все элементы должны быть в диапазоне [0, 1]
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("Все элементы матрицы переходов должны быть в диапазоне [0, 1]")
    
    # Проверка 3: Сумма элементов каждой строки должна быть равна 1
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        error_msg = "Сумма элементов каждой строки матрицы переходов должна быть равна 1.0:\n"
        for i, row_sum in enumerate(row_sums):
            error_msg += f"  Строка {i}: сумма = {row_sum:.10f}\n"
        raise ValueError(error_msg)
    
    return P

def validate_initial_vector(v0, n_states, tolerance=1e-6):
    """
    Проверка корректности вектора начального состояния
    """
    v0 = np.array(v0, dtype=float).flatten()
    
    # Проверка 1: Размерность должна соответствовать количеству состояний
    if v0.size != n_states:
        raise ValueError(f"Вектор начального состояния должен содержать {n_states} элементов. Получено: {v0.size}")
    
    # Проверка 2: Все элементы должны быть в диапазоне [0, 1]
    if np.any(v0 < 0) or np.any(v0 > 1):
        raise ValueError("Все элементы вектора начального состояния должны быть в диапазоне [0, 1]")
    
    # Проверка 3: Сумма элементов должна быть равна 1
    v_sum = v0.sum()
    if abs(v_sum - 1.0) > tolerance:
        raise ValueError(f"Сумма элементов вектора начального состояния должна быть равна 1.0. Получено: {v_sum:.10f}")
    
    return v0

def validate_cycles(m):
    """
    Проверка корректности количества циклов
    """
    if not isinstance(m, (int, np.integer)) or m <= 0:
        raise ValueError(f"Количество циклов должно быть положительным целым числом. Получено: {m}")
    
    return m

def check_probability_conservation(df, tolerance=1e-6):
    """
    Проверка сохранения суммы вероятностей на каждом шаге
    """
    all_valid = True
    for i in df.index:
        prob_sum = df.loc[i].sum()
        is_valid = abs(prob_sum - 1.0) < tolerance
        status = "✓" if is_valid else "✗"
        #print(f"Цикл {i:2d}: сумма = {prob_sum:.10f} {status}")
        if not is_valid:
            all_valid = False

    if not all_valid:
        print("⚠ ВНИМАНИЕ: Обнаружены отклонения в суммах вероятностей!")
    
    return all_valid

def check_convergence(df, last_n=3, tolerance=1e-6):
    """
    Проверка достижения стационарного состояния
    """
    if len(df) < last_n + 1:
        return False
    
    last_states = df.iloc[-last_n:].to_numpy()
    differences = np.abs(np.diff(last_states, axis=0))
    max_diff = differences.max()
    
    converged = max_diff < tolerance
    
    print(f"\n--- Проверка сходимости к стационарному состоянию ---")
    print(f"Максимальное изменение за последние {last_n} цикла: {max_diff:.10f}")
    
    if converged:
        print(f"✓ Система достигла стационарного состояния (изменение < {tolerance})")
    else:
        print(f"⚠ Система ещё не достигла стационарного состояния (изменение > {tolerance})")
    
    return converged

def simulate_markov(P, v0, m):
    # Валидация входных параметров
    P = validate_transition_matrix(P)
    n_states = P.shape[0]
    v0 = validate_initial_vector(v0, n_states)
    m = validate_cycles(m)
    
    # Моделирование
    v0 = v0.reshape(1, n_states)
    states = [v0.flatten()]
    v = v0.copy()
    
    for k in range(1, m+1):
        v = v @ P
        states.append(v.flatten())
    
    # Формирование DataFrame
    columns = [f's{i+1}' for i in range(n_states)]
    df = pd.DataFrame(states, columns=columns)
    df.index.name = 'Цикл'
    
    # Дополнительные проверки результатов
    check_probability_conservation(df)
    # check_convergence(df)
    
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
    try:
        P = np.array(DEFAULT_P, dtype=float)
        v0 = np.array(DEFAULT_V0, dtype=float).reshape(-1)
        
        df = simulate_markov(P, v0, DEFAULT_M)
        
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
        print("=" * 70)
        print(df.to_string(float_format=lambda x: f"{x:.6f}"))
        
        print("\n" + "-" * 70)
        final_state = df.loc[DEFAULT_M].to_numpy().round(6)
        print(f"Вектор состояния после выполнения {DEFAULT_M} циклов:")
        print(final_state)
        
        final_sum = df.loc[DEFAULT_M].to_numpy().sum().round(6)
        print(f"\nСумма вероятностей состояний: {final_sum}")

        
    except ValueError as e:
        print("\n" + "=" * 70)
        print("ОШИБКА ВАЛИДАЦИИ")
        print("=" * 70)
        print(f"❌ {e}")
        print("\nМоделирование прервано.")
    except Exception as e:
        print("\n" + "=" * 70)
        print("НЕПРЕДВИДЕННАЯ ОШИБКА")
        print("=" * 70)
        print(f"❌ {type(e).__name__}: {e}")