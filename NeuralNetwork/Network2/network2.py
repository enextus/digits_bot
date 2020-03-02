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

""" Раздел описаний """

""" Определение стоимостных функции """

# Определение среднеквадратичной стоимостной функции
class QuadraticCost(object):

    # Cтоимостная функция
    @staticmethod
    def fn(a, y): 
        return 0.5*np.linalg.norm(a-y)**2

    # Мера влияния нейронов выходного слоя на величину ошибки
    @staticmethod
    def delta(z, a, y): 
        return (a-y) * sigmoid_prime(z)

# Определение стоимостной функции на основе перекрестной энтропии
class CrossEntropyCost(object):
    
    # Cтоимостная функция
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    # Мера влияния нейронов выходного слоя на величину ошибки
    @staticmethod
    def delta(z, a, y):
        return (a-y)

""" Описание класса Network """
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes) # задаем количество слоев нейронной сети
        self.sizes = sizes # задаем список размеров слоев нейронной сети
        self.default_weight_initializer() # метод инициализации начальных весов связей и смещений по умолчанию
        self.cost=cost # задаем стоимостную функцию

    # метод инициализации начальных весов связей и смещений
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
        for x, y in zip(self.sizes[:-1], self.sizes[1:])] # задаем случайные начальные веса связей

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # задаем случайные начальные смещения
        self.weights = [np.random.randn(y, x)
        for x, y in zip(self.sizes[:-1], self.sizes[1:])] # задаем случайные начальные веса

    # the method counts the output signals of the neural network for given input signals
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) # computes the product of matrices
        return a

    # SGD
    #       lmbda = 0.0 # параметр сглаживания L2-регуляризации
    #     , evaluation_data=None # оценочная выборка
    #     , monitor_evaluation_cost=False # флаг вывода на экран информа-ции о значении стоимостной функции в процессе обучения, рассчитанном на оценочной выборке
    #     , monitor_evaluation_accuracy=False # флаг вывода на экран ин-формации о достигнутом прогрессе в обучении, рассчитанном на оценочной выборке
    #     , monitor_training_cost=False # флаг вывода на экран информации о значении стоимостной функции в процессе обучения, рассчитанном на обучающей выборке
    #     , monitor_training_accuracy=False # флаг вывода на экран инфор-мации о достигнутом прогрессе в обучении, рассчитанном на обучающей выборке
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

# Производная сигмоидальной функции
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

""" 
Конец раздела описаний
"""

"""
import os
os.chdir ('C:\\projects\\digits_bot\\NeuralNetwork\\Network2')
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)

"""


"""

(digits-env) PS C:\projects\digits_bot> python
Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:37:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import os
>>> os.chdir ('C:\\projects\\digits_bot\\NeuralNetwork\\Network2')
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
>>> import network2
>>> net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
>>> net.SGD(training_data, 30, 10, 0.5, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)
Epoch 0 training complete
--Cost on training data: 0.46336306112813624
--Accuracy on training data: 47180 / 50000
--Cost on evaluation data: 0.7659381061262848
--Accuracy on evaluation data: 9451 / 10000
Epoch 1 training complete
--Cost on training data: 0.49049852640338903
--Accuracy on training data: 47170 / 50000
--Cost on evaluation data: 0.8958750766326404
--Accuracy on evaluation data: 9406 / 10000
Epoch 2 training complete
--Cost on training data: 0.44066759950746875
--Accuracy on training data: 47630 / 50000
--Cost on evaluation data: 0.8993574892780825
--Accuracy on evaluation data: 9486 / 10000
Epoch 3 training complete
--Cost on training data: 0.42663977297701433
--Accuracy on training data: 47778 / 50000
--Cost on evaluation data: 0.9138976926147879
--Accuracy on evaluation data: 9530 / 10000
Epoch 4 training complete
--Cost on training data: 0.4240577946322011
--Accuracy on training data: 47875 / 50000
--Cost on evaluation data: 0.9353782893231701
--Accuracy on evaluation data: 9514 / 10000
Epoch 5 training complete
--Cost on training data: 0.42101247034650763
--Accuracy on training data: 47900 / 50000
--Cost on evaluation data: 0.9421021662472364
--Accuracy on evaluation data: 9528 / 10000
Epoch 6 training complete
--Cost on training data: 0.3941725447508594
--Accuracy on training data: 48223 / 50000
--Cost on evaluation data: 0.9212886398947676
--Accuracy on evaluation data: 9601 / 10000
Epoch 7 training complete
--Cost on training data: 0.4563447997870543
--Accuracy on training data: 47615 / 50000
--Cost on evaluation data: 1.0001777812834693
--Accuracy on evaluation data: 9488 / 10000
Epoch 8 training complete
--Cost on training data: 0.39455560561498804
--Accuracy on training data: 48264 / 50000
--Cost on evaluation data: 0.9490907076029371
--Accuracy on evaluation data: 9577 / 10000
Epoch 9 training complete
--Cost on training data: 0.40326702379449814
--Accuracy on training data: 48180 / 50000
--Cost on evaluation data: 0.9672822466195041
--Accuracy on evaluation data: 9550 / 10000
Epoch 10 training complete
--Cost on training data: 0.4028579769570545
--Accuracy on training data: 48227 / 50000
--Cost on evaluation data: 0.9627870878251531
--Accuracy on evaluation data: 9564 / 10000
Epoch 11 training complete
--Cost on training data: 0.41344738510902057
--Accuracy on training data: 48051 / 50000
--Cost on evaluation data: 0.9774468098658462
--Accuracy on evaluation data: 9559 / 10000
Epoch 12 training complete
--Cost on training data: 0.41202962278879635
--Accuracy on training data: 48194 / 50000
--Cost on evaluation data: 0.9791761024862768
--Accuracy on evaluation data: 9557 / 10000
Epoch 13 training complete
--Cost on training data: 0.41791735476978
--Accuracy on training data: 48061 / 50000
--Cost on evaluation data: 0.9969993190418316
--Accuracy on evaluation data: 9531 / 10000
Epoch 14 training complete
--Cost on training data: 0.41675799095126786
--Accuracy on training data: 48062 / 50000
--Cost on evaluation data: 0.9944257171175015
--Accuracy on evaluation data: 9515 / 10000
Epoch 15 training complete
--Cost on training data: 0.39941464918366465
--Accuracy on training data: 48192 / 50000
--Cost on evaluation data: 0.9847690510087352
--Accuracy on evaluation data: 9562 / 10000
Epoch 16 training complete
--Cost on training data: 0.381651450966655
--Accuracy on training data: 48369 / 50000
--Cost on evaluation data: 0.9637007826052297
--Accuracy on evaluation data: 9594 / 10000
Epoch 17 training complete
--Cost on training data: 0.36556770721319676
--Accuracy on training data: 48403 / 50000
--Cost on evaluation data: 0.9569530411865643
--Accuracy on evaluation data: 9614 / 10000
Epoch 18 training complete
--Cost on training data: 0.3829362769847325
--Accuracy on training data: 48272 / 50000
--Cost on evaluation data: 0.9665283027057314
--Accuracy on evaluation data: 9583 / 10000
Epoch 19 training complete
--Cost on training data: 0.37707103533173536
--Accuracy on training data: 48427 / 50000
--Cost on evaluation data: 0.9675845652723559
--Accuracy on evaluation data: 9573 / 10000
Epoch 20 training complete
--Cost on training data: 0.3691253949448845
--Accuracy on training data: 48451 / 50000
--Cost on evaluation data: 0.9560775340465058
--Accuracy on evaluation data: 9599 / 10000
Epoch 21 training complete
--Cost on training data: 0.38284935692240374
--Accuracy on training data: 48409 / 50000
--Cost on evaluation data: 0.9654351001986361
--Accuracy on evaluation data: 9586 / 10000
Epoch 22 training complete
--Cost on training data: 0.3618803883382167
--Accuracy on training data: 48522 / 50000
--Cost on evaluation data: 0.9506181676887475
--Accuracy on evaluation data: 9620 / 10000
Epoch 23 training complete
--Cost on training data: 0.3996338283341886
--Accuracy on training data: 48228 / 50000
--Cost on evaluation data: 0.9883998884013134
--Accuracy on evaluation data: 9557 / 10000
Epoch 24 training complete
--Cost on training data: 0.39803435703240025
--Accuracy on training data: 48260 / 50000
--Cost on evaluation data: 0.9885262394377196
--Accuracy on evaluation data: 9553 / 10000
Epoch 25 training complete
--Cost on training data: 0.40413335680935736
--Accuracy on training data: 48151 / 50000
--Cost on evaluation data: 0.9910974473215092
--Accuracy on evaluation data: 9539 / 10000
Epoch 26 training complete
--Cost on training data: 0.41491007748043796
--Accuracy on training data: 48035 / 50000
--Cost on evaluation data: 1.0078793280349707
--Accuracy on evaluation data: 9503 / 10000
Epoch 27 training complete
--Cost on training data: 0.3905182241035583
--Accuracy on training data: 48199 / 50000
--Cost on evaluation data: 0.9787794807047588
--Accuracy on evaluation data: 9547 / 10000
Epoch 28 training complete
--Cost on training data: 0.38625085208267596
--Accuracy on training data: 48284 / 50000
--Cost on evaluation data: 0.983212813792939
--Accuracy on evaluation data: 9578 / 10000
Epoch 29 training complete
--Cost on training data: 0.4215109798004295
--Accuracy on training data: 48111 / 50000
--Cost on evaluation data: 0.9972704678720175
--Accuracy on evaluation data: 9543 / 10000
([0.7659381061262848, 0.8958750766326404, 0.8993574892780825, 0.9138976926147879, 0.9353782893231701, 0.9421021662472364, 0.9212886398947676, 1.0001777812834693, 0.9490907076029371, 0.9672822466195041, 0.9627870878251531, 0.9774468098658462, 0.9791761024862768, 0.9969993190418316, 0.9944257171175015, 0.9847690510087352, 0.9637007826052297, 0.9569530411865643, 0.9665283027057314, 0.9675845652723559, 0.9560775340465058, 0.9654351001986361, 0.9506181676887475, 0.9883998884013134, 0.9885262394377196, 0.9910974473215092, 1.0078793280349707, 0.9787794807047588, 0.983212813792939, 0.9972704678720175], [9451, 9406, 9486, 9530, 9514, 9528, 9601, 9488, 9577, 9550, 9564, 9559, 9557, 9531, 9515, 9562, 9594, 9614, 9583, 9573, 9599, 9586, 9620, 9557, 9553, 9539, 9503, 9547, 9578, 9543], [0.46336306112813624, 0.49049852640338903, 0.44066759950746875, 0.42663977297701433, 0.4240577946322011, 0.42101247034650763, 0.3941725447508594, 0.4563447997870543, 0.39455560561498804, 0.40326702379449814, 0.4028579769570545, 0.41344738510902057, 0.41202962278879635, 0.41791735476978, 0.41675799095126786, 0.39941464918366465, 0.381651450966655, 0.36556770721319676, 0.3829362769847325, 0.37707103533173536, 0.3691253949448845, 0.38284935692240374, 0.3618803883382167, 0.3996338283341886, 0.39803435703240025, 0.40413335680935736, 0.41491007748043796, 0.3905182241035583, 0.38625085208267596, 0.4215109798004295], [47180, 47170, 47630, 47778, 47875, 47900, 48223, 47615, 48264, 48180, 48227, 48051, 48194, 48061, 48062, 48192, 48369, 48403, 48272, 48427, 48451, 48409, 48522, 48228, 48260, 48151, 48035, 48199, 48284, 48111])
>>>


"""

