
# baseline cnn model for mnist
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import time


# load train and test dataset
## input:
## output: 
## trainX.shape (60000, 28, 28, 1)
## trainY.shape (60000, 10)
def load_dataset():
	
	# load dataset
	
	trainX = np.load('drive/My Drive/ML/train_data.npy')
	trainy = np.load('drive/My Drive/ML/train_labels.npy')
	testX = np.load('drive/My Drive/ML/test_data.npy')
	testy = np.load('drive/My Drive/ML/test_labels.npy')
	trainX = trainX[:20001]
	trainy = trainy[:20001]
	testX = testX[:5001]
	testy = testy[:5001]

	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainy)
	testY = to_categorical(testy)
	return trainX, trainY, testX, testY


# scale pixels
## input: trainX, testX
## output: normalize greyScale to range(0,1)
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
## input: trainX, trainY
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation


  ## instantiation KFold() --> kfold
  ## kfold.split(trainX) --> [[index_list_for_train, index_list_for_test],[index_list_for_train, index_list_for_test],...]
  ## train[index_list_for_train] --> 48000 training pictures
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	start_time = time.time()
	print('start time:',start_time)
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	load_finished = time.time()
	print('data loading used:',load_finished-start_time)
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	normalize_finished = time.time()
	print('normalize used:',normalize_finished - load_finished)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	training_finished = time.time()
	print('training and evaluation used:',training_finished - normalize_finished)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()



