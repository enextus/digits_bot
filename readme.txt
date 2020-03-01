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



net = network.Network([784, 60, 10])
net.SGD(training_data, 60, 10, 2.0, test_data=test_data)
