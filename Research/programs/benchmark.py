import gc
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
