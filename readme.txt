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

# 784/16=49
net = network.Network([784, 49, 10])

1. # 784/14=56
net = network.Network([784, 56, 10])
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
Epoch 0: 458 / 10000
Epoch 1: 368 / 10000

Epoch 5: 693 / 10000
Epoch 6: 694 / 10000
Epoch 7: 750 / 10000

Epoch 15: 865 / 10000
Epoch 16: 710 / 10000
Epoch 17: 689 / 10000

Epoch 21: 946 / 10000
Epoch 22: 865 / 10000
Epoch 23: 915 / 10000

Epoch 49: 850 / 10000

2.
net = network.Network([784, 98, 10]) # 784/8=98
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)


>>> net = network.Network([784, 49, 10])
>>> net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
Epoch 0: 489 / 10000
Epoch 1: 560 / 10000
Epoch 2: 594 / 10000
Epoch 3: 548 / 10000
Epoch 4: 599 / 10000
Epoch 5: 577 / 10000
Epoch 6: 597 / 10000
Epoch 7: 584 / 10000

3. 
net = network.Network([784, 28, 10])
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)