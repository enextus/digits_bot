pip uninstall pytelegrambotapi
pip uninstall telebot

pip install telebot
pip install pytelegrambotapi

source /home/enextus/projects/digits_bot/digits-env/bin/activate

# start 

import os
os.chdir('/home/enextus/projects/digits_bot/NeuralNetwork/Network1')

os.chdir('C:\\projects\\digits_bot\\NeuralNetwork\\Network1')

.\Activate.ps1

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

net = network.Network([784, 49, 10]) # 784/16=49


1. 
net = network.Network([784, 56, 10]) # 784/14=56
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


>>> net = network.Network([784, 28, 10])
>>> net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
Epoch 0: 885 / 10000
Epoch 1: 904 / 10000
Epoch 2: 883 / 10000
Epoch 3: 902 / 10000
Epoch 4: 856 / 10000
Epoch 5: 898 / 10000
Epoch 6: 868 / 10000
Epoch 7: 825 / 10000
Epoch 8: 782 / 10000
Epoch 9: 819 / 10000
Epoch 10: 868 / 10000
Epoch 11: 843 / 10000
Epoch 12: 821 / 10000
Epoch 13: 834 / 10000
Epoch 14: 824 / 10000
Epoch 15: 832 / 10000
Epoch 16: 799 / 10000
Epoch 17: 804 / 10000
Epoch 18: 811 / 10000
Epoch 19: 837 / 10000
Epoch 20: 804 / 10000
Epoch 21: 814 / 10000
Epoch 22: 790 / 10000
Epoch 23: 789 / 10000
Epoch 24: 766 / 10000
Epoch 25: 773 / 10000
Epoch 26: 796 / 10000
Epoch 27: 765 / 10000
Epoch 28: 813 / 10000
Epoch 29: 761 / 10000
>>> 


>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 1.0, test_data=test_data)
Epoch 0: 250 / 10000
Epoch 1: 245 / 10000
Epoch 2: 187 / 10000
Epoch 3: 180 / 10000
Epoch 4: 253 / 10000
Epoch 5: 190 / 10000
Epoch 6: 217 / 10000
Epoch 7: 188 / 10000


>>> net = network.Network([784, 49, 10])
>>> net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
Epoch 0: 91 / 10000
Epoch 1: 283 / 10000
Epoch 2: 427 / 10000
Epoch 3: 408 / 10000
Epoch 4: 393 / 10000
Epoch 5: 409 / 10000
Epoch 6: 483 / 10000
Epoch 7: 393 / 10000
Epoch 8: 437 / 10000
Epoch 9: 491 / 10000
Epoch 10: 500 / 10000
Epoch 11: 429 / 10000