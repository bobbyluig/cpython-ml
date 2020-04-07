import gc
import logging
import time
import os
import threading

from flask import Flask, g

app = Flask(__name__)

# Some large data that is in memory.
_data = [[] for _ in range(1000000)]

# Counter for number of requests.
count = 0

# Global lock.
lock = threading.Lock()


@app.after_request
def after_request(response):
    global count

    with lock:
        count += 1

    return response


def reward():
    global count

    # Running average.
    average = 0
    alpha = 0.01

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

        # Ran out of memory.
        if gc.memory_usage() > 130000000:
            r = -50
        else:
            r /= (time.time() - start)

        # Compute running average of reward.
        if average is None:
            average = r
        else:
            average = alpha * r + (1 - alpha) * average

        # Print out statistics.
        print('Average Reward: {:<10.6f} {}'.format(average, gc.memory_usage()), end='\r')

        # Provide reward.
        gc.reward(r)

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
        os.exit(0)
