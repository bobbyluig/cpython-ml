import gc


def do_a_thing():
    a = {}
    for i in range(1000):
        a[i] = []
    a = None


print(do_a_thing())
print(do_a_thing())
print(do_a_thing())
print(do_a_thing())

print(gc.print_tuning_stats())
