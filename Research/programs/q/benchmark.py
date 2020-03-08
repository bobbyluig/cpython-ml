import gc
import time


def benchmark(operation, reward, target_iterations=20, alpha=0.1):
    # Start benchmarking.
    print('Obtaining baseline.')

    # Compute target time.
    times = []
    for _ in range(target_iterations):
        start = time.time()
        operation()
        times.append(time.time() - start)
    target = sum(times) / len(times)

    # Baseline complete.
    print('Baseline complete.')

    # Moving average.
    average = None

    # Main measurement loop.
    while True:
        # Get the number of collections.
        gc_stats = gc.get_stats()
        start_collections = [gc_stats[i]['collections'] for i in range(3)]

        # Get the starting time.
        start = time.time()

        # Perform operation with GC.
        gc.enable()
        operation()
        gc.disable()

        # Compute the delta of time.
        delta_time = time.time() - start

        # Compute the delta of collections.
        gc_stats = gc.get_stats()
        delta_collections = [gc_stats[i]['collections'] - start_collections[i] for i in range(3)]

        # Provide a reward based on time.
        r = reward(delta_time)

        # Compute running average of reward.
        if average is None:
            average = r
        else:
            average = alpha * r + (1 - alpha) * average

        # Provide reward.
        gc.reward(1000 + r)

        # Print out statistics.
        # print('\033c', end='')
        print('Target Time: {:<10.6f}'.format(target))
        # print('Current Time: {:<10.6f}'.format(delta_time))
        print('Average Reward: {:<10.6f}'.format(average))
        # print('Collections: ({:<6} {:<6} {:<6})'.format(*delta_collections))


def collect(operation, reward):
    # Zero time.
    zero_time = None

    # Main measurement loop.
    while True:
        # Get the starting time.
        start = time.time()

        # Perform operation with GC.
        gc.enable()
        operation()
        gc.disable()

        # Compute the delta of time.
        current_time = time.time()
        delta_time = current_time - start

        # Compute memory usage.
        memory = gc.memory_usage()

        # Provide a reward based on time.
        r = reward(delta_time, memory)
        gc.reward(r)

        # Set zero time.
        if zero_time is None:
            zero_time = current_time

        # Time since start.
        time_since_start = current_time - zero_time
        if time_since_start > 60 * 5:
            break

        # Print out statistics.
        print('{},{},{}'.format(time_since_start, r, memory))
