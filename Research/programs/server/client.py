import threading
import time

import requests

lock = threading.Lock()
i = 0
target = 1000


def run():
    global i

    while True:
        with lock:
            if i == target:
                return
            i += 1
            print('{0:.3f}'.format(i / target), end='\r')

        requests.get('http://localhost:5000/')


threads = []
start = time.time()

for _ in range(128):
    t = threading.Thread(target=run)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(target / (time.time() - start))
