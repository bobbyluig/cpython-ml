import gc
import logging

from flask import Flask

app = Flask(__name__)

# Some large data that is in memory.
_data = [[] for _ in range(1000000)]

# Function collect/don't collect
# After each return False, get stats about mem usage
# After each return True, gets stats about mem usage and success


# 108.79300753927068/35.338132730735744
count = 0
MANUAL_GC = True


@app.route('/')
def hello_world():
    global count

    if MANUAL_GC:
        if count % 80 == 0:
            gc.collect(2)
        count += 1

    # Allocate some objects to trigger garbage collection a few times (and once on generation 2).
    data = []
    for _ in range(70000):
        data.append([])

    # Return some response.
    return 'Hello World!'


if __name__ == '__main__':
    if MANUAL_GC:
        gc.disable()
    else:
        gc.set_threshold(700, 10, 10)
        # gc.set_debug(gc.DEBUG_STATS)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run()
