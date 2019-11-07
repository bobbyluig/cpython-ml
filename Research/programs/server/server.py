import gc
import logging

from flask import Flask

app = Flask(__name__)


# Build big lists, do gc
# When Python GC is invoked now
# Get program counter, current memory pressure
# Do perfect gc

# 1) Create web server of requests comes in, build noncyclic, reply to request
# 2) Implement version with perfect gc vs imperfect gc
# 3) Find what information is available in GC (before/after), number of cycles still alive, time since last collection,
# and how long ago in allocation, PC at last collection
# 4) Find a way to get these to run continuously and measure data
# 5) LRU cache is lookup time
# 6) TXN throughput
# 7) Run program for a fixed amount of time (like 2 minutes)

@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    # gc.disable()
    gc.set_threshold(700, 10, 10)
    # gc.set_debug(gc.DEBUG_STATS)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run()
