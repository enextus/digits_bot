"""
network2.py
A module for creating and training a neural network for handwritten digit 
recognition based on the stochastic gradient descent method for 
a direct neural network and cost function based on cross entropy, 
regularization and an improved method of initializing neural network weights.

Group: <Specify Group Number>
Name: <Indicate the name of the student>
"""

#### Библиотеки

# Стандартные библиотеки
import json # библиотека для кодирования/декодирования данных/объектов Python
import random # библиотека функций для генерации случайных значений
import sys # библиотека для работы с переменными и функциями, имеющими отношение к интерпретатору и его окружению

# Сторонние библиотеки
import numpy as np # библиотека функций для работы с матрицами

""" ---Раздел описаний--- """

""" -- Определение стоимостных функции --"""

class QuadraticCost(object): # Определение среднеквадратичной стоимостной функции

    @staticmethod
    def fn(a, y): # Cтоимостная функция
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y): # Мера влияния нейронов выходного слоя на величину ошибки
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object): # Определение стоимостной функции на основе перекрестной энтропии

    @staticmethod
    def fn(a, y): # Cтоимостная функция
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y): # Мера влияния нейронов выходного слоя на величину ошибки
        return (a-y)

""" --Описание класса Network--"""
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes) # задаем количество слоев нейронной сети
        self.sizes = sizes # задаем список размеров слоев нейронной сети
        self.default_weight_initializer() # метод инициализации начальных весов связей и смещений по умолчанию
        self.cost=cost # задаем стоимостную функцию

    def default_weight_initializer(self): # метод инициализации начальных весов связей и смещений
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
        for x, y in zip(self.sizes[:-1], self.sizes[1:])] # задаем случайные начальные веса связей

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x)
        for x, y in zip(self.sizes[:-1], self.sizes[1:])] # задаем случайные начальные веса

    def feedforward(self, a): # the method counts the output signals of the neural network for given input signals
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) # computes the product of matrices
        return a



    # lmbda = 0.0 # параметр сглаживания L2-регуляризации
    #     , evaluation_data=None # оценочная выборка
    #     , monitor_evaluation_cost=False # флаг вывода на экран информа-ции о значении стоимостной функции в процессе обучения, рассчитанном на оценочной выборке
    #     , monitor_evaluation_accuracy=False # флаг вывода на экран ин-формации о достигнутом прогрессе в обучении, рассчитанном на оценочной выборке
    #     , monitor_training_cost=False # флаг вывода на экран информации о значении стоимостной функции в процессе обучения, рассчитанном на обучающей выборке
    #     , monitor_training_accuracy=False # флаг вывода на экран инфор-мации о достигнутом прогрессе в обучении, рассчитанном на обучающей выборке
    #     ):
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, evaluation_data=None, monitor_evaluation_cost=False, monitor_evaluation_accuracy=False, monitor_training_cost=False, monitor_training_accuracy=False):
        
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)
        training_data = list(training_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print ("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("--Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("--Accuracy on training data: {} / {}".format(accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("--Cost on evaluation data: {}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("--Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy


    # Шаг градиентного спуска
    #
    #     self                # указатель на объект класса
    #     , mini_batch        # подвыборка
    #     , eta               # скорость обучения
    #     , lmbda             # параметр сглаживания L2-регуляризации
    #     , n                 #
    def update_mini_batch(self, mini_batch, eta, lmbda, n):

        nabla_b = [np.zeros(b.shape) for b in self.biases] # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights] # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # послойно вычисляем градиенты dC/db и dC/dw для текущего прецедента (x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # суммируем градиенты dC/db для различных прецедентов текущей подвыборки
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # суммируем градиенты dC/dw для различных прецедентов текущей подвыборки
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] # обновляем все веса w нейронной сети
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] # обновляем все смещения b нейронной сети


    # Алгоритм обратного распространения
    #     self # Указатель на объект класса
    #      , x # Вектор входных сигналов
    #      , y # Ожидаемый вектор выходных сигналов
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases] # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights] # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)

        # Определение переменных
        activation = x # Выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activations = [x] # Список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        zs = [] # Список активационных потенциалов по всем слоям (первоначально пуст)

        # Прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # Считаем активационные потенциалы текущего слоя
            zs.append(z) # Добавляем элемент (активационные потенциалы слоя) в конец списка
            activation = sigmoid(z) # Считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activations.append(activation) # Добавляем элемент (выходные сигналы слоя) в конец списка
        # Обратное распространение
        delta = (self.cost).delta(zs[-1], activations[-1], y) # Считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        nabla_b[-1] = delta # Градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())# Градиент dC/dw для слоя L (BP4)

        for l in range(2, self.num_layers):
            z = zs[-l] # Активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            sp = sigmoid_prime(z) # Считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp # Считаем меру влияния нейронов l-го слоя на величину ошибки (BP2)
            nabla_b[-l] = delta # Градиент dC/db для l-го слоя (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w) # Градиент dC/dw для l-го слоя (BP4)

    # Оценка прогресса в обучении
    #     self # Указатель на объект класса
    #     , data # Набор данных (выборка)
    #     , convert=False # Признак необходимости изменять формат представления результата работы нейронной сети
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    # Значение функции потерь по всей выборке
    #     self # Указатель на объект класса
    #     , data # Набор данных (выборка)
    #     , lmbda # Параметр сглаживания L2-регуляризации
    #     , convert=False # Признак необходимости изменять формат представления результата работы нейронной сети
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        data = list(data)
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)

        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost

    # Запись нейронной сети в файл
    def save(self, filename): 
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()





# Загрузка нейронной сети из файла
def load(filename): 

    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

# Определение сигмоидальной функции активации
def sigmoid(z): 
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): # Производная сигмоидальной функции
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

""" --- Конец раздела описаний--- """

"""

import os
os.chdir ('C:\\projects\\digits_bot\\NeuralNetwork\\Network2')
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

"""

