import threading

import requests

lock = threading.Lock()
i = 0
target = 5000


def run():
    global i

    while True:
        with lock:
            if i == target:
                return
            i += 1
            print('{0:.3f}'.format(i / 5000), end='\r')

        requests.get('http://localhost:5000/')


threads = []

for _ in range(128):
    t = threading.Thread(target=run)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
