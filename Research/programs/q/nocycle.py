from benchmark import benchmark


def operation():
    for i in range(200):
        a = []
        for _ in range(7000):
            a.append([])


# GC Information
# Instructions: 4
if __name__ == '__main__':
    benchmark(operation, lambda time: -time)
