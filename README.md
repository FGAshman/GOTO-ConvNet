# GOTO-ConvNet
Source code for the Convolutional Neural Network to classify astronomical objects using the GOTO data. 4th year Astro project at University of Sheffield


Code requires use of the Galaxy Zoo 2 database (can be downloaded from: https://data.galaxyzoo.org/) in order for survey matching and pre-labelling, and must be in CSV format.

Initial file structure requires the raw images to be downloaded to a folder named '142' that saves the cutouts created from source_detection.py into a folder named '142_cutouts' with individual folders for 'stars', 'all_galaxies', and also 'elliptical_galaxies' and 'spiral_galaxies'. The ids, pixel positions, RA and Dec, and galaxy types for each source is saved in a table for each raw image saved to a folder '142_data'.

Cutouts will be saved as 30x30 pixel fits images in the '142_cutouts' folder, where augmentations (rotation of specified angles, and flipping of the images) can be performed to achieve balanced data sets using the rotate-image.py file. Once the data sets required for training have equal numbers they will be compiled into a pickle file used to input into the ConvNet by the create_training_data.py file.

The model is then trained, and tested using the ConvNet_model.py file, and will produce a confusion matrix, and examples of 25 incorrectly labelled images.

