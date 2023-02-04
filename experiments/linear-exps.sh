#! /bin/bash

python ../src/experiments.py --layers=0 --l2=0 --l2_diff=0 \
    --name=baseline_linear --epochs=100 --optim=LBFGS --lr=1

python ../src/experiments.py --layers=0 --l2=0.1 --l2_diff=0.1 \
    --name=linear_regularized --epochs=100 --optim=LBFGS --lr=1