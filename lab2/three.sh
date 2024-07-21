num_runs=5

for ((run=1; run<=num_runs; run++)); do
    echo "Running training (Run $run/$num_runs)"
    python trainer.py
done