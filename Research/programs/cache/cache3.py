import functools
import gc
import random
import time


class LLNode:
    def __init__(self, previous, next, value):
        self.previous = previous
        self.next = next
        self.value = value


@functools.lru_cache(maxsize=8)
def create_ll(size):
    node = LLNode(None, None, 0)

    for i in range(1, size - 1):
        new_node = LLNode(node, None, i)
        node.next = new_node
        node = new_node

    return node


def operation():
    for i in range(100):
        size = random.randint(2, 500)
        create_ll(size)


if __name__ == '__main__':
    try:
        while True:
            random.seed(0)

            start_collections = gc.get_stats()[2]['collections']
            start = time.time()
            operation()
            delta = time.time() - start
            delta_collections = gc.get_stats()[2]['collections'] - start_collections

            print(delta, delta_collections)
            gc.reward(0)
    except KeyboardInterrupt:
        pass

    gc.collect()
    gc.disable()

    import matplotlib.pyplot as plt
    import numpy as np

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
