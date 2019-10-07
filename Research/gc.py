import functools
import random
import time

# import gc
# gc.set_debug(gc.DEBUG_STATS)

# 1000 iterations @ 10.508163928985596
# gc.set_threshold(700, 10, 10)

# 1000 iterations @ 8.501218795776367
# gc.set_threshold(20000, 10, 10)

# 1000 iterations @ 8.43212103843689
# gc.disable()

CACHE_SIZE = 16
LINE_COUNT = 2048


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2


def ccw(a: Point, b: Point, c: Point) -> bool:
    return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)


def intersect(l1: Line, l2: Line):
    return ccw(l1.p1, l2.p1, l2.p2) != ccw(l1.p2, l2.p1, l2.p2) and \
           ccw(l1.p1, l1.p2, l2.p1) != ccw(l1.p1, l1.p2, l2.p2)


@functools.lru_cache(maxsize=CACHE_SIZE)
def count_intersections(seed):
    random.seed(seed)

    lines = []
    for _ in range(LINE_COUNT):
        p1 = Point(random.random(), random.random())
        p2 = Point(random.random(), random.random())
        line = Line(p1, p2)
        lines.append(line)

    test_line = lines.pop()

    intersections = []
    for line in lines:
        if intersect(test_line, line):
            intersections.append((test_line, line))

    return intersections


start = time.time()

for seed in range(100):
    count_intersections(seed)

print(time.time() - start)