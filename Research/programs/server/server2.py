import gc
import logging
import os
import socketserver
import threading

import requests
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


lock = threading.Lock()
i = 0
port = 0
target = 1500


def run():
    global i
    global port

    while True:
        with lock:
            if i == target:
                gc.print_tuning_stats()
                os._exit(0)

            i += 1

        requests.get('http://localhost:{}/'.format(port))


def send_request():
    threads = []

    for _ in range(32):
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


original_socket_bind = socketserver.TCPServer.server_bind


def socket_bind_wrapper(self):
    global port

    ret = original_socket_bind(self)
    port = self.socket.getsockname()[1]
    socketserver.TCPServer.server_bind = original_socket_bind

    t = threading.Thread(target=send_request)
    t.start()

    return ret


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'

    socketserver.TCPServer.server_bind = socket_bind_wrapper
    app.run(port=0)
