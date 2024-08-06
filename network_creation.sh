#!/bin/bash

python network.py -m fcnn -d mnist
python network.py -m fcnn -d fmnist
python network.py -m fcnn -d kmnist
python network.py -m fcnn -d emnist
echo "Finished fcnn models"

python network.py -m lenet5 -d mnist
python network.py -m lenet5 -d fmnist
python network.py -m lenet5 -d kmnist
python network.py -m lenet5 -d emnist
echo "Finished lenet5 models"

python network.py -m resnet18 -d cifar10
python network.py -m resnet18 -d cifar100
python network.py -m resnet18 -d svhn
echo "Finished resnet18 models"

python network.py -m rnn -d imdb
python network.py -m rnn -d reuters
echo "Finished rnn models"

echo "Finished creating models"