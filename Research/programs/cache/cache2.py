import functools
import random
import gc


class LLNode:
    def __init__(self, previous, next, value):
        self.previous = previous
        self.next = next
        self.value = value


@functools.lru_cache(maxsize=16)
def create_ll(size):
    node = LLNode(None, None, 0)

    for i in range(1, size - 1):
        new_node = LLNode(node, None, i)
        node.next = new_node
        node = new_node

    return node


if __name__ == '__main__':
    random.seed(0)

    for i in range(2000):
        size = random.randint(2, 500)
        create_ll(size)

    gc.print_tuning_stats()
