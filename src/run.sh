#!/usr/bin/env bash

source ../venv/bin/activate
for ((i = 1; i <= 3; i++)); do
  echo "Running iteration $i"
  python3 run-optuna.py &
done

wait $(jobs -p)