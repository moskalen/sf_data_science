import numpy as np

number = np.random.randint(0, 101)
count = 0

while True:
    count += 1
    predict_number = int(input("Guess a number from 1 to 100: "))

    if predict_number > number:
        print("too big")
    elif predict_number < number:
        print("too small")
    else:
        print(f"Yes! The number {number} was guessed from {count} attempts.")
        break
