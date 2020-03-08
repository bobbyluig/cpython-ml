import requests
import threading


def run():
    while True:
        requests.get('http://localhost:5000/')


for _ in range(64):
    t = threading.Thread(target=run)
    t.start()
