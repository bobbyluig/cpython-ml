import gc
import time


def collect(operation, reward):
    # Zero time.
    zero_time = None

    # Get the starting time.
    start = time.time()

    # Main measurement loop.
    while True:
        # Get the starting random actions.
        random_start = gc.random_actions()

        # Perform operation with GC.
        gc.enable()
        operation()
        gc.disable()

        # Compute the delta of time.
        current_time = time.time()
        delta_time = current_time - start

        # Get the starting time.
        start = time.time()

        # Compute memory usage.
        memory = gc.memory_usage()

        # Provide a reward based on time.
        r = reward(delta_time, memory)
        gc.reward(r)

        # Show how the policy is evolving over time.
        gc.print_policy()

        # Set zero time.
        if zero_time is None:
            zero_time = current_time

        # Time since start.
        time_since_start = current_time - zero_time
        if time_since_start > 60 * 10:
            break

        # Change in number of actions.
        actions = gc.random_actions() - random_start

        # Print out statistics.
        print('{},{},{},{}'.format(time_since_start, r, memory, actions))
