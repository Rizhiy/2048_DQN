import random
from time import sleep

from twenty_forty_eight_python.puzzle import GameGrid

KEYS = ["'w'", "'s'", "'a'", "'d'"]


def random_press(grid):
    sleep(0.1)
    key = random.choice(KEYS)
    status = grid.key_down(key)
    if status == -1:
        print("YOU LOSE")
        print("Final score: {}".format(grid.score))
        exit()
    grid.after(0, random_press(grid))


if __name__ == '__main__':
    while True:
        grid = GameGrid()
        grid.after(0, random_press(grid))
        grid.mainloop()
