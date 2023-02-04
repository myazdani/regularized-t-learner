#! /bin/bash

l2_regs="0 0.0001 0.001 0.01 0.1"
l2_diffs="0 0.0001 0.001 0.01 0.1"

for l2_reg in $l2_regs; do
  for l2_diff in $l2_diffs; do
    python ../src/experiments.py --layers=0 --l2=$l2_reg --l2_diff=$l2_diff \ 
        --name=baseline_linear --max_steps=1 --optim=LBFGS --lr=1
  done
done
