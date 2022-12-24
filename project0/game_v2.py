"""Gues a number game
Computer creates and tries to guess a number
"""
import numpy as np


def random_predict(number: int = 1) -> int:
    """Guess a number randomly

    Args:
        number (int, optional): A number which should be guessed. Defaults to 1.

    Returns:
        int: the number of attempts
    """
    count = 0

    while True:
        count += 1
        predicted_number = np.random.randint(1, 101)
        if number == predicted_number:
            break

    return count


def score_game(predict_func) -> int:
    """Average attemps from 1000 to guess a number

    Args:
        predict_func (type): guessing func

    Returns:
        int: averate attempts
    """
    count_ls = []
    np.random.seed(1)
    random_array = np.random.randint(1, 101, size=1000)

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls))
    print(f"In average it guesses a number from {score} attempts")
    return score


if __name__ == '__main__':
    score_game(random_predict)
