import gc
import logging
import time

from flask import Flask, g

app = Flask(__name__)

# Some large data that is in memory.
_data = [[] for _ in range(1000000)]

# Running average.
average = 0
alpha = 0.01


@app.before_request
def before_request():
    g.request_start_time = time.time()


@app.after_request
def after_request(response):
    global average

    # Compute reward
    r = -(time.time() - g.request_start_time)

    # Compute running average of reward.
    if average is None:
        average = r
    else:
        average = alpha * r + (1 - alpha) * average

    # Print out statistics.
    print('Average Reward: {:<10.6f}'.format(average), end='\r')

    # Provide reward.
    gc.reward(r)

    return response


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

    app.run()
