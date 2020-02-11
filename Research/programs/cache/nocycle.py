import gc
import random
import time


def operation():
    for i in range(200):
        a = []
        for _ in range(7000):
            a.append([])


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
    print('Target:', sum(times) / 20)

    # Moving average.
    alpha = 0.1
    average = None

    while True:
        random.seed(0)

        start_collections = gc.get_stats()[2]['collections']
        start = time.time()
        operation()
        delta = time.time() - start
        delta_collections = gc.get_stats()[2]['collections'] - start_collections

        if gc.memory_usage() > 64000000:
            delta += 100000

        if average is None:
            average = delta
        else:
            average = alpha * delta + (1 - alpha) * average

        print('Current: {:<10.6f} Collections: {:<10}'.format(average, delta_collections), end='\r')
        gc.reward(-delta)
