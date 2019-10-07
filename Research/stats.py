from collections import namedtuple
import sys

Entry = namedtuple('Entry', ['time', 'generation', 'objects', 'bytes_before', 'elapsed', 'bytes_after'])


def analyze(filename):
    with open(filename) as f:
        raw_trace = f.readlines()

    trace = []
    for i in range(0, len(raw_trace), 7):
        entry = Entry(
            time=float(raw_trace[0][9:]),
            generation=int(raw_trace[1][26:27]),
            objects=[int(count) for count in raw_trace[2][32:].split()],
            bytes_before=[int(memory) for memory in raw_trace[3][38:].split()],
            elapsed=float(raw_trace[5].split(',')[-1].strip().split()[0][:-1]),
            bytes_after=[int(memory) for memory in raw_trace[6][37:].split()],
        )
        trace.append(entry)

    total_gc_time = sum(entry.elapsed for entry in trace)
    print('Total GC time: {}s'.format(total_gc_time))

    average_memory_collected = sum(sum(entry.bytes_before) - sum(entry.bytes_after) for entry in trace) / len(trace)
    print('Average Memory Collected: {}kB'.format(average_memory_collected / 1024))


if __name__ == '__main__':
    analyze(sys.argv[1])