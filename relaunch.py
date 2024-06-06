MAIN_NOF_EPOCHS = 0
MAIN_DO_INIT = True

import subprocess
import time
import sys


# Valid: файлы с 1-го по 5-й плюс (10_861 - 5) случайно выбранных
# Train: файлы с 6-го по 10-й плюс (180_000 - 5) оставшихся


def interval(a, b, step=1):
    return range(a, b+1, step) if step > 0 else range(a, b-1, step)


def end(seq):
    return len(seq) - 1


DELAY = 10
try:
    NOF_EPOCHS = int(sys.argv[1])
except:
    NOF_EPOCHS = MAIN_NOF_EPOCHS
try:
    DO_INIT = bool(int(sys.argv[2]))
except:
    DO_INIT = MAIN_DO_INIT



print('\nBEGIN SUBPEROCESSING ROUTINE\n')

if DO_INIT:
    print(f'\nBEGIN INITIALIZATION\n')
    subprocess.run(['python3', 'learnlearn.py', '0', '1'])
    time.sleep(DELAY)
for epoch in interval(1, NOF_EPOCHS):
    print(f'\nBEGIN EPOCH {epoch}\n')
    subprocess.run(['python3', 'learnlearn.py', str(epoch), '0'])
    time.sleep(DELAY)
