"""Игра угадай число
Компьютер сам загадывает и сам угадывает число
"""

import numpy as np


def guess_number(number: int = 1) -> int:
    """Отгадываем число уменьшая промежуток, где число может находиться

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0
    left_border = 1
    right_border = 100

    while True:
        count += 1
        predict_number = left_border + (right_border - left_border) // 2

        if number == predict_number:
            break

        if predict_number < number:
            left_border = (
                predict_number
                if left_border != predict_number
                else left_border + 1
            )
        else:
            right_border = (
                predict_number
                if right_border != predict_number
                else right_border - 1
            )

    return count


def score_game(guess_func) -> int:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(1000))

    for number in random_array:
        count_ls.append(guess_func(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за:{score} попыток")
    return score


if __name__ == "__main__":
    # RUN
    score_game(guess_number)
