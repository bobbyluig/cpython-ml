import gc
import random
import time


def benchmark(operation, reward, target_iterations=10, alpha=0.1):
    # Compute target time.
    times = []
    for _ in range(target_iterations):
        start = time.time()
        operation()
        times.append(time.time() - start)
    print('Target:', sum(times) / len(times))

    # Moving average.
    average = None

    # Main measurement loop.
    while True:
        start = time.time()
        operation()
        delta = time.time() - start

        r = reward(delta)

        if average is None:
            average = r
        else:
            average = alpha * r + (1 - alpha) * average

        print('Current: {:<10.6f}'.format(average), end='\r')
        gc.reward(r)


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


def operation():
    for i in range(200):
        simulate(100, 2)


if __name__ == '__main__':
    benchmark(operation, lambda time: -time)
