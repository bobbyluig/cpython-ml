import functools
import gc
import random
import time
from benchmark import benchmark


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


# GC Information
# Instructions: 10
if __name__ == '__main__':
    # Warm up
    random.seed(0)
    operation()

    # Reward function.
    def reward(t):
        if gc.memory_usage() > 32000000:
            t += 1000

        return -t

    # Benchmark.
    benchmark(operation, reward)

