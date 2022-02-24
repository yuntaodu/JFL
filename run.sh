#!/bin/bash

# mnist - mnist_m
python run.py -s mnist -t mnist_m -cuda 0 -logdir /runs/JFL -modeldir /models/JFL -data_root /root/dataset_DA >results/mnist_mnist_m.log  2>&1

# mnist - svhn
# python run.py -s mnist -t svhn -cuda 0 >results/mnist_svhn.log 2>&1

# svhn - mnist
# python run.py -s svhn -t mnist -cuda 0 >results/svhn_mnist.log 2>&1

# digits - svhn
# python run.py -s digits -t svhn -cuda 0 >results/digits_svhn.log 2>&1
