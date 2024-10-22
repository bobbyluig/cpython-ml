import gc
import random
import time


class Transaction:
    def __init__(self):
        self.start_time = time.time()
        self.dependencies = set()
        self.data = [0] * 1000


def dfs(start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in state.dependencies:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def simulate(a, b):
    aborted = set()
    transactions = []

    for i in range(random.randint(0, a)):
        transactions.append(Transaction())

    for transaction in transactions:
        for _ in range(random.randint(0, b)):
            transaction.dependencies.add(random.choice(transactions))

    cycles = set(frozenset([node] + path) for node in transactions for path in dfs(node, node))

    for cycle in cycles:
        to_abort = max(cycle, key=lambda t: t.start_time)
        aborted.add(to_abort)

    return aborted


if __name__ == '__main__':
    random.seed(0)

    for i in range(1000):
        simulate(100, 2)

    gc.print_tuning_stats()