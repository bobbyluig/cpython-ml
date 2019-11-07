import functools
import gc
import random
import time

# 2.81040096282959/3.5310161113739014
count = 0
MANUAL_GC = False


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
    if MANUAL_GC:
        gc.disable()
    else:
        gc.set_threshold(700, 10, 10)
        # gc.set_debug(gc.DEBUG_STATS)

    random.seed(0)

    start = time.time()

    for i in range(10000):
        print('{0:.3f}'.format(i / 10000), end='\r')
        size = random.randint(2, 500)
        create_ll(size)

        if MANUAL_GC:
            if count % 172 == 0:
                gc.collect(2)
            count += 1

    print(time.time() - start)
