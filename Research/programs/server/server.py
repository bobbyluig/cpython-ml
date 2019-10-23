import gc
import logging

from flask import Flask

app = Flask(__name__)


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

