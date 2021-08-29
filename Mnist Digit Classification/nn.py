## name: Zhoukaong Xia
## netID: zxx180009

###########################################################################################################

## overall structure:

## data import and preprocessing (line 20 to line 110)

## build a class called Layer: mainly used to contain weights and bias (line 125 to line 200)

## build a class called NeuralNetwork: a neural newwork (line 200 to line 380)
    ## functions of NeuralNetwork:
    ## 1. NeuralNetwork.add_layer(layer_object), add new layers to the nn.
        ## example: nn.add_layer(Layer(1000, 100, 'sigmoid'))
    ## 2. NeuralNetwork.train(train_data, train_labels, learning_rate, max_epochs):
        ## param train_data: shape is (60000,784)
        ## param train_labels: shape is (60000,10) reshape done in data preprocessing
        ## param learning_rate: choose from 0 to 1
        ## param max_epochs: choose from 10 to 60
        
## testing (line 390 to line 410)
    ## final accuracy: could reach 94.18% after 50 epoches (cost 10 minutes on google colab)
    
## display (line 410 to line 420)

import os.path
import urllib.request
import gzip
import numpy             as np
import matplotlib.pyplot as plt
import time

DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS


################################################################################
#
# DATA preprocessing
#
################################################################################
START_TIME = time.time()
# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)


# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)
tl = test_labels.copy()

train_data = train_data.reshape(train_data.shape[0],784)
train_data /= 255
test_data = test_data.reshape(test_data.shape[0],784)
test_data /= 255


## categorize labels
rows = len(train_labels)
columns = 10
temp = np.zeros((rows,columns),dtype = int)
for i in range(rows):
    temp[i,train_labels[i]] = 1
train_labels = temp
  
rows = len(test_labels)
columns = 10
temp = np.zeros((rows,columns),dtype = int)
for i in range(rows):
    temp[i,test_labels[i]] = 1
test_labels = temp

#########################################################

## Two classes: class1: NeuralNetwork ; class2: Layer

#########################################################

## a hidden layer or output in our neural network.
class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """

        :param int n_input: The input size 

        :param int n_neurons: The number of neurons in this layer.

        :param str activation: The activation method to use.

        :param weights: The layer's weights.

        :param bias: The layer's bias.

        """
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)-0.5
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)-0.5
        self.last_activation = None
        self.error = None
        self.delta = None


    def activate(self, x):

        r = np.dot(x, self.weights) + self.bias
        
        self.last_activation = self._apply_activation(r)
        
        '''
        print('X',X)
        print('self.weights',self.weights)
        print('r',r)
        print('self.last_activation',self.last_activation)
        '''
        
        return self.last_activation
        
    ## apply activation method accoding to self.activation
    def _apply_activation(self, r):

        # self.activation == None
        if self.activation is None:

            return r

        # tanh
        if self.activation == 'tanh':

            return np.tanh(r)

        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    
    ## derivative actication function
    def apply_activation_derivative(self, r):


        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        return r


## neural network with several layers
class NeuralNetwork:

    def __init__(self):
        self._layers = []
        
        ## for each training epoch, record the test accuracy
        self.accuracy_record = []
        

    ## use the prev class layer as input
    def add_layer(self, layer):

        self._layers.append(layer)

    ## input the training data X 
    ## output the forward output of neural network
    def feed_forward(self, X):

        for layer in self._layers:
            
            X = layer.activate(X)
        
        return X

    
    def predict(self, X):
        """

        Predicts a class

        :param X: The input values.

        :return: The predictions of class, format like 8

        """

        ff = self.feed_forward(X)
        
        # One row

        if ff.ndim == 1:
            return np.argmax(ff)

        # Multiple rows

        return np.argmax(ff, axis=1)
        

    def backpropagation(self, X, y, learning_rate):
        """

        Performs the backward propagation algorithm and updates the layers weights.

        :param X: The input values as an single image flatted to a vector

        :param y: The label, format like [0,0,0,0,0,0,0,0,0,1]

        :param float learning_rate: The learning rate (between 0 and 1).

        """

        # Feed forward for the output

        output = self.feed_forward(X)
        ## print('X',X,'output',output)
        # Loop over the layers backward

        ## for each layer:
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:

                ####
                layer.error = y - output
                # The output = layer.last_activation in this case
                
                ## output_deltas = outputs * (1 - outputs) * (outputs - target)
                layer.delta = layer.error * layer.apply_activation_derivative(output)
                '''
                print('y',y,'output',output)
                print('error',y-output)
                print('layer.apply_activation_derivative(output)',layer.apply_activation_derivative(output))
                print('layer.delta',layer.delta)
                '''
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
                '''
                print('y',y,'output',output)
                print('error',y-output)
                print('layer.apply_activation_derivative(output)',layer.apply_activation_derivative(output))
                print('layer.delta',layer.delta)
                '''

        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)

            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)

            ## layer.weights += [1*3] * [3*1] *
            layer.weights += layer.delta * input_to_use.T * learning_rate
            
            '''
            print('layer.delta',layer.delta)
            print('*',layer.delta * input_to_use.T * learning_rate)
            print('layer.weights',layer.weights)
            '''

    def train(self, train_data, train_labels, learning_rate, max_epochs):
        """

        Trains the neural network using backpropagation.

        :param X: The input values.

        :param y: The target values.

        :param float learning_rate: The learning rate (between 0 and 1).

        :param int max_epochs: The maximum number of epochs (cycles).

        :return: The list of calculated MSE errors.

        """
        training_start = time.time()
        mses = []
        ## empty the accuracy_record
        self.accuracy_record = []
        batch_size = 1000
        if max_epochs * batch_size>60000-1:
            max_epochs =int( 60000 / batch_size) - 1
        for i in range(max_epochs):
            
            X = train_data[i*batch_size:(i+1)*batch_size]
            y = train_labels[i*batch_size:(i+1)*batch_size]
            '''
            ## sum the 20 x together and 20 y together input it as 
            X = np.sum(X,axis = 0) 
            X = X/batch_size
            y = np.sum(y,axis = 0)
            y = y/batch_size
            print(y)
            '''
            for j in range(len(X)):
                '''
                print('X[j]',X[j],'y[j]',y[j])
                '''
                self.backpropagation(X[j], y[j], learning_rate)

            mse = np.mean(np.square(y - nn.feed_forward(X)))
            mses.append(mse)
            mse = float(mse)
            mse = round(mse,6)
            
            t = round((time.time() - training_start), 1)
            
            a = self.accuracy(test_data,tl) *100
            a = round(a,2)
            
            self.accuracy_record.append(a)
            
            print(f'Epoch {i}({t}s): Training MSE = {mse}, Test Accuracy: {a}%')
    
    
    def accuracy(self, test_data, test_labels):
        L = len(test_data)
        true = 0
        false = 0
        for i in range(L):
            if self.predict(test_data[i]) == test_labels[i]:
                true+=1
            else:
                false+=1
        return true/(true+false)
    
    
########################################################
##   
##     Testing
##
########################################################

nn = NeuralNetwork()
## add layers
## first layer added is the first hidden layer
## nn.add_layer(Layer(prev_input_num, nodes, 'tanh'))
nn.add_layer(Layer(784, 1000, 'tanh'))
nn.add_layer(Layer(1000, 100, 'sigmoid'))
nn.add_layer(Layer(100, 10, 'sigmoid'))

# Train the neural network
nn.train(train_data, train_labels, 0.05, 50)

# Plot Accuracy vs Epoch 
plt.plot(nn.accuracy_record)
plt.title('Test Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.show()


fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ## predict label by nn
    pred = nn.predict(test_data[i])
    ax[-1].set_title('True: ' + str(tl[i]) + ' NN: ' + str(pred))
    plt.imshow(img, cmap='Greys')
plt.show()




