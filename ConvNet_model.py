import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils.multiclass import unique_labels
import pickle
import time, pdb

#Loading in the dataset
pickle_in = open("X_star-galaxy.pickle","rb")
features = pickle.load(pickle_in)

pickle_in = open("y_star-galaxy.pickle","rb")
labels = pickle.load(pickle_in)

#Visualising results code taken from https://github.com/kylepob/planeClassifier and adapted for our purposes:
def visualise_incorrect_labels(X, y_real, y_predicted):
	count = 0
	figure = plt.figure()
	incorrect_label = (y_real != y_predicted)
	y_real = y_real[incorrect_label]
	y_predicted = y_predicted[incorrect_label]
	X = X[incorrect_label, :, :, :]
	X = np.squeeze(X, axis = -1)

	maximum_square = np.ceil(np.sqrt(X.shape[0]))

	for i in range(25):
		count += 1
		figure.add_subplot(5, 5, count)
		plt.imshow(X[i, :, :], cmap = 'gray', origin = 'lower')
		plt.axis('off')
		plt.title("Predicted: " + str(int(y_predicted[i])) + ", Real: " + str(int(y_real[i])), fontsize = 10)
	plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


'''
Including metrics for precision, recall, and f_beta. Taken from the original keras documentation which was removed from later versions.
Taken from: https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7
'''
def precision(y_true, y_pred):
	true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_pos/(predicted_pos + K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_pos/(possible_pos + K.epsilon())
	return recall

def fbeta_score(y_true, y_pred, beta = 1):
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision)')

	#If there are no true positives, fix the F score at 0 like sklearn
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta**2
	fbeta_score = (1 + bb)*(p*r)/(bb*p + r + K.epsilon())
	return fbeta_score

def ConvNet_model(X, y, n_layers, model_name, class_names, Epochs = 16, SAVE_TEST_DATA = False):
	#Define hyperparameters:
	Kernel = (3, 3)
	min_neurons = 64
	max_neurons = 256

	#Define the number of neurons in each convolution layer:
	steps = np.floor(max_neurons/(n_layers + 1))
	neurons = np.arange(min_neurons, max_neurons, steps)
	neurons = neurons.astype(np.int32)

	#Applying a name to the model for tensorboard:
	NAME = "conv-star-galaxy-normalised-{}".format(int(time.time()))

	#Splitting the data into training and test data:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

	#Adding in a line to optionally save the test data
	if SAVE_TEST_DATA == True:
		pickle_out = open("X_star-galaxy-test.pickle", "wb")
		pickle.dump(X_test, pickle_out)
		pickle_out.close()

		pickle_out = open("y_star-galaxy-test.pickle", "wb")
		pickle.dump(y_test, pickle_out)
		pickle_out.close()


	#Define a model:
	model = Sequential()

	#Add convolution layers:
	for i in range(0, n_layers):
		if i == 0:
			#The input layer
			model.add(Conv2D(neurons[i], Kernel, input_shape = X_train.shape[1:]))
		else:
			#Every other layer
			model.add(Conv2D(neurons[i], Kernel))

		model.add(Activation('relu'))
		#model.add(MaxPooling2D(pool_size=(2, 2)))

	#Adding a dropout layer to reduce overfitting
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(max_neurons))
	model.add(Activation('relu'))

	#Add output layer:
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	#Compile the model:
	model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              		metrics = ['accuracy', fbeta_score, precision, recall])

	#Print a summary of the model:
	model.summary()

	#Applying callbacks to the model:
	#Early stopping callback to prevent further training once a chosen quantity stops improving
	Patience = 5
	early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = Patience,
								   mode = 'auto', restore_best_weights = True)
	#Tensorboard callback to visualise the progress of the model
	tensorboard = TensorBoard(log_dir = "logs/{}".format(NAME))

	callbacks = [early_stopping, tensorboard]

	#Fitting the model
	model.fit(X_train, y_train, batch_size = 64, epochs = 16, verbose = 2, validation_split = 0.2,
          		callbacks = callbacks, shuffle = True)

	model.save(model_name)


	#Testing the model, creating a confusion matrix, and visualising the incorrect labels
	test_predictions = model.predict_classes(X_test)


	print("Accuracy on test data = {}".format(accuracy_score(y_test, test_predictions)))

	plot_confusion_matrix(y_test, test_predictions, classes = class_names, normalize = True, title = 'Normalised confusion matrix')
	plt.show()

	visualise_incorrect_labels(X_test, np.asarray(y_test), np.asarray(test_predictions).ravel())

ConvNet_model(features, labels, 6, 'models/conv_star_galaxy_metric_test.model', ['Non-extended', 'Extended'])
