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

python network.py -m resnet10 -d cifar10
python network.py -m resnet10 -d svhn
python network.py -m resnet10 -d caltech-101
echo "Finished resnet10 models"

python network.py -m alexnet -d mnist
python network.py -m alexnet -d fmnist
python network.py -m alexnet -d kmnist
python network.py -m alexnet -d emnist
python network.py -m alexnet -d cifar10
python network.py -m alexnet -d svhn
python network.py -m alexnet -d caltech-101
echo "Finished alexnet models"

python network.py -m mobilenetv2 -d cifar10
python network.py -m mobilenetv2 -d svhn
python network.py -m mobilenetv2 -d caltech-101
echo "Finished mobilenetv2 models"

python network.py -m vggnet16 -d mnist
python network.py -m vggnet16 -d fmnist
python network.py -m vggnet16 -d kmnist
python network.py -m vggnet16 -d emnist
python network.py -m vggnet16 -d cifar10
python network.py -m vggnet16 -d svhn
python network.py -m vggnet16 -d caltech-101
echo "Finished vggnet16 models"

echo "Finished creating models"