import gc
import matplotlib.pyplot as plt
import numpy as np
import random

for _ in range(10000000):
    x = random.random()
    y = random.random()

    if 0.4 < x < 0.5 and 0.2 < y < 0.7:
        output = (1, -1)
        gc.ann_train((x, y), output)
    else:
        output = (-1, 1)

    gc.ann_train((x, y), output)

print('Done training!')

xs = np.linspace(0, 1, num=50)
ys = np.linspace(0, 1, num=50)

for x in xs:
    for y in ys:
        no_collect, collect = gc.ann_evaluate((x, y))

        if no_collect > collect:
            color = 'red'
        else:
            color = 'green'

        plt.plot(x, y, marker='o', color=color, markersize=2)

plt.xlabel('Instruction')
plt.ylabel('Memory')
plt.savefig('cache3.png')


for _ in range(2):
    gc.ann_train((1.0, 2.0), (1.0, 0.5))