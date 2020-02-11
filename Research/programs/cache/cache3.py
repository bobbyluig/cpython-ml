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

    # Don't collect here!
    a = []
    for _ in range(700):
        a.append([])

    for i in range(1, size - 1):
        new_node = LLNode(node, None, i)
        node.next = new_node
        node = new_node

    return node


def operation():
    for i in range(500):
        size = random.randint(2, 500)
        create_ll(size)


if __name__ == '__main__':
    # Warm up
    random.seed(0)
    operation()

    # Compute target.
    times = []
    for _ in range(20):
        random.seed(0)
        start = time.time()
        operation()
        times.append(time.time() - start)
    print('Target', sum(times) / 20)

    # Moving average.
    alpha = 0.1
    average = None

    while True:
        random.seed(0)

        start = time.time()
        operation()
        delta = time.time() - start

        if gc.memory_usage() > 64000000:
            delta += 100000

        if average is None:
            average = delta
        else:
            average = alpha * delta + (1 - alpha) * average

        print('Current', average, end='\r')
        gc.reward(-delta)

