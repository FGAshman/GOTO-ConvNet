import numpy as np 
import matplotlib.pyplot as plt
import astropy
from astropy.utils.data import download_file
from astropy.io import fits
import os
import cv2
import tensorflow as tf
from astropy.modeling.models import Rotation2D
from scipy import misc
from scipy.ndimage import rotate
from scipy.misc import imrotate, imsave
import os, pandas, time, random
import shutil, pdb
from astropy.visualization import ImageNormalize, MinMaxInterval

def move_files(path, destination, rand_numb):
	'''
	Moves a random selection of files from a selected path to another selected folder (folder must already be created)
	Arguements: 
		path: Directory which contains the folders which are to be moved
		destination: Directory which the folders are to moved to
		rand_numb: the proportion of files which are moved during the function

	'''
	files = os.listdir(path)
	print("Number of files in the path folder:", len(files))
	src_files = np.random.choice(os.listdir(path), int(len(files)*rand_numb))
	print("Number of files in the path folder after choosing a random number of files:", len(src_files))

	for file_name in src_files:
		full_file_name = os.path.join(path, file_name)
		if (os.path.isfile(full_file_name)):
			shutil.move(full_file_name, destination)

#move_files("C:/Users/fredd/OneDrive/Documents/Coding/Python/Machine-learning-project/Project - code/142_cutouts/Stars", 
#	"C:/Users/fredd/OneDrive/Documents/Coding/Python/Machine-learning-project/Project - code/142_cutouts/stars-overflow", 0.9)




#Not sure if I need to do the splitting up, can do it within tensorflow

DATADIR = "C:/Users/fredd/OneDrive/Documents/Coding/Python/Machine-learning-project/Project-code/142_cutouts"
CATEGORIES = ["stars", "all_galaxies"]

training_data = [] 

def create_training_data():
	start = time.time()
	print("Creating the training set")
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			hdul = fits.open(os.path.join(path,img))
			array = hdul[0].data
			mean_array = np.mean(array)
			std_array = np.std(array)
			img_array = (array - mean_array)/std_array
			training_data.append([array, class_num])	
	print("Finished creating the training set in:", time.time() - start)


create_training_data()
#print(training_data.shape)

#randomising the data
random.shuffle(training_data)

#X is the feature set, y is the labels
X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)

#Convert X to a numpy array, where the 1 is due to it being greyscale
X = np.array(X).reshape(-1, 30, 30, 1)

#saving the training data
import pickle
pickle_out = open("X_star-galaxy-final.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_star-galaxy-final.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()



