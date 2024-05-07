#!/bin/bash
num_runs=5
for i in $(seq 1 $num_runs); do
  python -u "/home/dxtvzw/hse/ecal-optimization/src/main.py"
done
