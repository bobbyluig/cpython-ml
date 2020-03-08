import gc
import logging
import time
import sys
import threading

from flask import Flask, g

app = Flask(__name__)

# Some large data that is in memory.
_data = [[] for _ in range(1000000)]

# Counter for number of requests.
count = 0

# Counter for GC.
gc_count = 0

# Global lock.
lock = threading.Lock()

# Zero time.
zero_time = None


@app.after_request
def after_request(response):
    global count

    with lock:
        count += 1

    return response


def reward():
    global count, zero_time

    # The last number of request served.
    last_count = count

    # The starting time.
    start = time.time()

    while True:
        # Wait a bit.
        time.sleep(1)

        with lock:
            if count == 0:
                continue

            r = (count - last_count)
            last_count = count

        # Get current time.
        current_time = time.time()

        # Get memory usage.
        memory = gc.memory_usage()

        # Ran out of memory.
        if memory > 130000000:
            gc.collect()
            r = -200
        else:
            r /= (current_time - start)

        # Provide reward.
        gc.reward(r)

        # Set zero time.
        if zero_time is None:
            zero_time = current_time

        # Time since start.
        time_since_start = current_time - zero_time
        if time_since_start > 60 * 5:
            raise Exception

        # Print out statistics.
        print('{},{},{}'.format(time_since_start, r, memory))

        # Start time.
        start = time.time()


@app.route('/')
def hello_world():
    # Allocate some objects to trigger garbage collection a few times (and once on generation 2).
    data = []
    for _ in range(70000):
        data.append([])

    # Return some response.
    return 'Hello World!'


# GC Information
# Instructions: 202
if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    threading.Thread(target=reward).start()

    try:
        app.run()
    except KeyboardInterrupt:
        raise Exception
