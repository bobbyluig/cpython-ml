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

            start = time.time()
            operation()
            if gc.memory_usage() > 50000000:
                gc.collect()
                print('force collection')
                delta = 10000
            else:
                delta = time.time() - start

            r = 1000 / (1 + delta)
            print(r)
            gc.reward(r)
    except KeyboardInterrupt:
        pass
