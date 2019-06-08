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
import os, pandas, time

def galaxy_rotate(angle):
	#Reading in data from the original csv file (which was used to create cutouts)
	path = "C:/Users/fredd/OneDrive/Documents/Coding/Python/Machine-learning-project/Project - code/142_cutouts/spiral_galaxies_original"
	i = 0
	for img in os.listdir(path):
		i += 1
		#for i in range(len(os.listdir(path))):
		hdul = fits.open(os.path.join(path,img))
		hdr = hdul[0].header
		img = hdul[0].data
		ra = hdr['RA']
		dec = hdr['Dec']
		#Performing the flipping
		start = time.time()
		data_rot = imrotate(img, angle, interp = 'bilinear')
		hdu = fits.PrimaryHDU(data_rot)
		hdu_header = hdul[0].header #get header (empty atm)
		hdu.header.set('RA', ra) #set header of each cutout to have RA and DEC
		hdu.header.set('DEC', dec)
		hdu.writeto('142_cutouts/spiral_galaxies/spiral_rotate{0}_{1}.fits'.format(angle, i), overwrite = True)
		print('Created rotated fits files for image {0} in time: {1}'.format(i, (time.time() - start)))
angle = [1, -1, 2, -2, 3, -3, 90, 270]
for i in angle:
	galaxy_rotate(i)

def galaxy_flip():
	#Reading in data from the original csv file (which was used to create cutouts)
	path = "C:/Users/fredd/OneDrive/Documents/Coding/Python/Machine-learning-project/Project - code/142_cutouts/spiral_galaxies"
	i = 0
	for img in os.listdir(path):
		i += 1
		#for i in range(len(os.listdir(path))):
		hdul = fits.open(os.path.join(path,img))
		hdr = hdul[0].header
		img = hdul[0].data
		ra = hdr['RA']
		dec = hdr['Dec']
		#Performing the flipping
		start = time.time()
		data_flipud = np.flipud(img)
		hdu = fits.PrimaryHDU(data_flipud)
		hdu_header = hdul[0].header #get header (empty atm)
		hdu.header.set('RA', ra) #set header of each cutout to have RA and DEC
		hdu.header.set('DEC', dec)
		hdu.writeto('142_cutouts/spiral_galaxies/spiral_flipud{0}.fits'.format(i), overwrite = True)

		data_fliplr = np.fliplr(img)
		hdu = fits.PrimaryHDU(data_fliplr)
		hdu_header = hdul[0].header #get header (empty atm)
		hdu.header.set('RA', ra) #set header of each cutout to have RA and DEC
		hdu.header.set('DEC', dec)
		hdu.writeto('142_cutouts/spiral_galaxies/spiral_fliplr{0}.fits'.format(i), overwrite = True)

		print('Created flipped fits files for image {0} in time: {1}'.format(i, (time.time() - start)))

galaxy_flip()



