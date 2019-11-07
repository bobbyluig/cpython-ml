import gc

gc.disable()
gc.collect()

def do_a_thing():
    before = gc.get_memory()

    a = {}
    for i in range(10000):
        a[i] = []
    a = None

    return gc.get_memory()

print(do_a_thing())
print(do_a_thing())
print(do_a_thing())
print(do_a_thing())