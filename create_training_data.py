import numpy as np 
import astropy, os, time, random, pickle
from astropy.io import fits

################################# FUNCTIONS #################################

def create_training_data(DATADIR, CATEGORIES):
	'''
	Compiles the images into a list that can be fed into the machine learning algorithm. Also includes normalization of the
	images and randomises the list to aid in training the algorithm

	Arguments:
		DATADIR: Path to the where the image data is located
		CATEGORIES: A list of the different labels of objects to be trained on i.e. Galaxies and Stars
	'''
	training_data = [] 

	start = time.time()
	print("Creating the training set")
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			hdul = fits.open(os.path.join(path,img))
			array = hdul[0].data

			# Performing z-scale normalization on the images
			mean_array = np.mean(array)
			std_array = np.std(array)
			img_array = (array - mean_array)/std_array
			training_data.append([array, class_num])

	print(f"Finished creating the training set in: {time.time() - start}")

	# Randomising the data
	random.shuffle(training_data)

	return training_data

def save_pickle(file_path, file):
	'''
	Simple helper function to save any file as a pickle file

	Arguments:
		file_path: Path to where the pickle file will be saved
		file: The actual file to be saved
	'''
	
	pickle_out = open(file_path, "wb")
	pickle.dump(file, pickle_out)
	pickle_out.close()


def data_to_pickle(training_data, FEATURE_SAVE_PATH, LABEL_SAVE_PATH, IMG_SIZE = 30):
	'''
	Converting the training data to a pickle file to be fed into the machine learning

	Arguments:
		training_data: List of randomised, normalized image data to be saved and used in the algorithm
		FEATURE_SAVE_PATH: Save path of the feature data set
		LABEL_SAVE_PATH: Save path of the label data set

	Optional Arguments:
		IMG_SIZE: x and y dimensions of the images - which end up as the size of the first layer of the CNN
				--- Defaults to 30  pixels 
	'''

	# X is features (the images), y are labels (what type of object it is)
	X = []
	y = []

	for features, label in training_data:
		X.append(features)
		y.append(label)

	#Convert X to a numpy array, where the 1 is due to it being greyscale
	X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

	# Saving the training data as pickle files
	save_pickle(FEATURE_SAVE_PATH, X)
	save_pickle(LABEL_SAVE_PATH, y)


################################# USAGE #################################

# Creating the training data
training_data = create_training_data(DATADIR, CATEGORIES)
