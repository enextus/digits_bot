pip uninstall pytelegrambotapi
pip uninstall telebot

pip install telebot
pip install pytelegrambotapi

source /home/enextus/projects/digits_bot/digits-env/bin/activate

# start 

import os
os.chdir('/home/enextus/projects/digits_bot/NeuralNetwork/Network1')
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network



net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

1. >>> net = network.Network([784, 56, 10])
>>> net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
Epoch 0: 458 / 10000
Epoch 1: 368 / 10000
Epoch 2: 471 / 10000
Epoch 3: 624 / 10000