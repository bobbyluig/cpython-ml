import gc
import logging
import time

from flask import Flask

app = Flask(__name__)

# Some large data that is in memory.
_data = [[] for _ in range(1000000)]

# Function collect/don't collect
# After each return False, get stats about mem usage
# After each return True, gets stats about mem usage and success


# 108.79300753927068/35.338132730735744
average = 0
alpha = 0.01


@app.route('/')
def hello_world():
    global average

    start = time.time()

    # Allocate some objects to trigger garbage collection a few times (and once on generation 2).
    data = []
    for _ in range(70000):
        data.append([])

    delta = time.time() - start
    gc.reward(1 / (1 + delta))

    if average is None:
        average = delta
    else:
        average = alpha * delta + (1 - alpha) * average

    print('Current', average, end='\r')

    # Return some response.
    return 'Hello World!'


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run()
