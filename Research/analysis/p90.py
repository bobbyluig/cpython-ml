import heapq
import sys

count = int(sys.argv[1])
file = sys.argv[2]
heap_size = int(round(count * 0.1))
heap = []

with open(file) as f:
    for _ in range(count):
        x = f.readline()
        x = int(x)
        if len(heap) < heap_size:
            heapq.heappush(heap, x)
        else:
            heapq.heappushpop(heap, x)

print(heapq.heappop(heap))