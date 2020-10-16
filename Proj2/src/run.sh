#!/usr/bin/env bash


for batch_size in 1 5 10 20 50 100 200; do
    for learning_rate in  1e-2 1e-3 1e-4 1e-5 1e-6; do
	python3 test.py --lr=$learning_rate  --batch_size=$batch_size --epochs=500 --nb_run=10
    done
done
