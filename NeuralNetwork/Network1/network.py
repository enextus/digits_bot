#!/usr/local/bin/python
# -*- coding: utf-8 -*-

"""
A module for creating and training a neural network for recognizing handwritten numbers using the gradient descent method.
Group: Python 3
Name: Eduard
"""

"""
Import of built-in modules
"""
import os, sys # system tools;
import random # random integers generation;

"""
Import of third-party modules
"""
import numpy as np # for matrix calculation;

"""
Class Network
"""
class Network(object): # used to describe a neural network;
    def __init__(self, sizes): # class konstruktor; self - a pointer to the class object; sizes - list of sizes of all neural layers;
        self.num_layers = len(sizes) # set the number of layers of the neural network;
        self.sizes = sizes # set list of sizes of neural network layers;
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # set random initial offsets
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # set random initial bond weights (RATINGS)

    def feedforward(self, a): # the method counts the output signals of the neural network for given input signals
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) # computes the product of matrices
            return a
    
    # Stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data): # self - a pointer to the class object; training_data - training sample; epochs - number of training eras; mini_batch_size - sample size; eta - learning speed; test_data - test sample
        test_data = list(test_data) # create a list of test sample objects
        n_test = len(test_data) # calculate the length of the test sample
        training_data = list(training_data) # create a list of training sample objects
        n = len(training_data) # calculate the size of the training sample
        for j in range(epochs): # cycle by epochs
            random.shuffle(training_data) # mix the elements of the training sample
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)] #create subsamples
            for mini_batch in mini_batches: # loop through subsamples
                self.update_mini_batch(mini_batch, eta) # one step gradient descent
            print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)) # look at the progress in learning

    # Step gradient descent
    def update_mini_batch(self, mini_batch, eta): # self - a pointer to the class object; subsample; learning speed; 
        nabla_b = [np.zeros(b.shape) for b in self.biases] # list of dC / db gradients for each layer (initially filled with zeros)
        nabla_w = [np.zeros(w.shape) for w in self.weights] # list of dC / db gradients for each layer (initially filled with zeros)

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # compute the dC / db and dC / dw gradients layer by layer for the current use case (x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # summarize dC / db gradients for various use cases of the current subsample
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # summarize dC / dw gradients for various use cases of the current subsample
        
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] # we update all the weight "w" of the neural network
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)] # we update all the weight "b" of the neural network

    # Backpropagation algorithm
    def backprop(self, x, y): # pointer to the class object; vector of input signals; expected output vector
        nabla_b = [np.zeros(b.shape) for b in self.biases] # list of dC / db gradients for each layer (initially filled with zeros)
        nabla_w = [np.zeros(w.shape) for w in self.weights] # list of dC / dw gradients for each layer (initially filled with zeros)
        
        # определение переменных
        activation = x # выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activations = [x] # список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        zs = [] # список активационных потенциалов по всем слоям (первоначально пуст)
        
        # прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # считаем активационные потенциалы текущего слоя
            zs.append(z) # добавляем элемент (активационные потенциалы слоя) в конец списка
            activation = sigmoid(z) # считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activations.append(activation) # добавляем элемент (выходные сигналы слоя) в конец списка

        # обратное распространение
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        nabla_b[-1] = delta # градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # градиент dC/dw для слоя L (BP4)

        for l in range(2, self.num_layers):
            z = zs[-l] # активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            sp = sigmoid_prime(z) # считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # считаем меру влияния нейронов l-го слоя на величину ошибки (BP2)
            nabla_b[-l] = delta # градиент dC/db для l-го слоя (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())# градиент dC/dw для l-го слоя (BP4)
        return (nabla_b, nabla_w)

    # Оценка прогресса в обучении
    def evaluate(self, test_data): 
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # Вычисление частных производных стоимостной функции по выходным сигналам последнего слоя
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

# definition of sigmoidal activation function
def sigmoid(z): 
    return 1.0/(1.0+np.exp(-z))

# Производная сигмоидальной функции
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))



"""
program start
"""
net = Network([2, 3, 1]) # create a neural network of three layers
"""
program end
"""

""" Displays the result on the screen: """
print('Network net:')
print('Layers amount:', net.num_layers)
for i in range(net.num_layers):
    print('Amount of neurons in the layer', i,':',net.sizes[i])
for i in range(net.num_layers-1):
    print('W_',i+1,':')
    print(np.round(net.weights[i],2))
    print('b_',i+1,':')
    print(np.round(net.biases[i],2))
