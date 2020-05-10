for learningRate in '0.001' '0.01'
do
  for ((i = 1; i <= 20; i++))
  do
    # Transaction program.
    echo "Transaction $learningRate $i"
    Q_EPSILON_START=0.5 Q_EPSILON_END=0.0001 Q_FENCE_MEMORY=26214400 Q_LEARNING_RATE="$learningRate" \
     ../python3/bin/python3.8 -u Research/programs/q/transaction.py > "Research/data/transaction1_${learningRate}_${i}.txt"

    # Cache program.
    echo "Cache $learningRate $i"
    Q_EPSILON_START=0.5 Q_EPSILON_END=0.00005 Q_FENCE_MEMORY=26214400 Q_LEARNING_RATE="$learningRate" \
     ../python3/bin/python3.8 -u Research/programs/q/cache.py > "Research/data/cache1_${learningRate}_${i}.txt"

    # Web server.
    echo "Server $learningRate $i"
    $(sleep 5; python3 Research/programs/q/client.py) &
    Q_EPSILON_START=0.5 Q_EPSILON_END=0.00001 Q_FENCE_MEMORY=146800640 Q_LEARNING_RATE="$learningRate" \
     ../python3/bin/python3.8 -u Research/programs/q/server_collect.py > "Research/data/server1_${learningRate}_${i}.txt"

    # Reset.
    killall -9 python3
  done
done