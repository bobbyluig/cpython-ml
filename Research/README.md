```
rm -f /dev/shm/temp; PYTHONTRACEMALLOC=1 ../python3/bin/python3.7 Research/programs/cache/cache.py 2>> /dev/shm/temp; python3 Research/analysis/p90.py $(tail -n 1 /dev/shm/temp) /dev/shm/temp; rm /dev/shm/temp

rm -f /dev/shm/temp; ../python3/bin/python3.7 Research/programs/cache/cache.py 2>> /dev/shm/temp; python3 Research/analysis/gc.py /dev/shm/temp; rm /dev/shm/temp
```