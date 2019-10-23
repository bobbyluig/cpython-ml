import functools
import gc
import random


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


if __name__ == '__main__':
    # gc.disable()
    gc.set_threshold(700, 10, 10)
    # gc.set_debug(gc.DEBUG_STATS)

    random.seed(0)

    for i in range(2000):
        print('{0:.3f}'.format(i / 2000), end='\r')
        size = random.randint(2, 100)
        create_ll(size)
