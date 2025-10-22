# Корреляционная функция

По определению:

$$
R_Y(t_1, t_2) = \mathbb{E}[Y(t_1)Y(t_2)] = \mathbb{E}[e^{-X t_1} e^{-X t_2}] = \mathbb{E}[e^{-X (t_1 + t_2)}].
$$

То есть нужно вычислить математическое ожидание от экспоненты случайной величины:

$$
R_Y(t_1, t_2) = \int_0^{\infty} e^{-x (t_1 + t_2)} f_X(x) \, dx = \int_0^{\infty} e^{-x (t_1 + t_2)} \lambda e^{-\lambda x} \, dx.
$$

Объединим экспоненты:

$$
R_Y(t_1, t_2) = \lambda \int_0^{\infty} e^{-x (\lambda + t_1 + t_2)} \, dx = \lambda \left[ \frac{1}{\lambda + t_1 + t_2} \right] = \frac{\lambda}{\lambda + t_1 + t_2}.
$$

# Нормированная корреляционная функция

Нормированная корреляционная функция определяется как:

$$
\rho_Y(t_1, t_2) = \frac{R_Y(t_1, t_2)}{\sqrt{R_Y(t_1, t_1) R_Y(t_2, t_2)}}.
$$

Сначала найдём корреляционную функцию:

$$
R_Y(t, t) = \frac{\lambda}{\lambda + 2t}.
$$

Тогда:

$$
\rho_Y(t_1, t_2) =
\frac{\lambda / (\lambda + t_1 + t_2)}{\sqrt{(\lambda / (\lambda + 2t_1))(\lambda / (\lambda + 2t_2))}}
= \frac{\lambda}{\lambda + t_1 + t_2} \cdot \frac{\sqrt{(\lambda + 2t_1)(\lambda + 2t_2)}}{\lambda}.
$$

Упрощаем:

$$
\rho_Y(t_1, t_2) =
\frac{\sqrt{(\lambda + 2t_1)(\lambda + 2t_2)}}{\lambda + t_1 + t_2}.
$$