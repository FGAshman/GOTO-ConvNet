import numpy as np 
import astropy, os, time
from astropy.io import fits
from scipy.misc import imrotate, imsave

################################# FUNCTIONS #################################

def data_from_fits_files(hdul):
	'''
	Retrieves data from the fits files

	Arguments:
		hdul: Object with the data that the fits file contains - will primarily be called using fits.open()
	'''
	hdr = hdul[0].header
	img = hdul[0].data
	ra = hdr['RA']
	dec = hdr['Dec']

	return img, ra, dec

def write_fits_from_data(img_data, folder_path, base_file_name, RA, DEC):
	'''
	Helper function to save image data as fits files, and write the RA and DEC to the header

	Arguments:
		img_data: Raw Image data to be saved in the fits file
		folder_path: Path to the main folder which the images will be stored in
		base_file_name: path that the individual fits files are saved - allows for iteration
		RA: Right Ascension of the object as stated in the header of the original fits file
		DEC: Declination of the object as stated in the header of the original fits file
	'''

	hdu = fits.PrimaryHDU(img)
	hdu.header.set('RA', ra)
	hdu.header.set('DEC', dec)

	hdu.writeto(folder_path + base_file_name, overwrite = True)




def fits_image_augment(file_path, base_file_name, angle, ANGLE_ROTATE, VERTICAL_FLIP):
	'''
	Augments the number of fits images by creating rotated versions and flipped versions of an original image
	Allows for the option to create either through a boolean values

	Arguments:
		file_path: The file path of the base folder of the images
		base_file_name: The path to the base file name of the original images before being augmented
		angle: A list of angles to rotate the original images through
		ANGLE_ROTATE: Boolean value to determine whether to augment the data through rotations
		VERTICAL_FLIP: Boolean value to determine whether to flip the images vertically (up-down) or horizontally (left-right)
	'''


	#Reading in data from the original csv file (which was used to create cutouts)
	i = 0
	for img in os.listdir(file_path):

		# Opening up the fits files and retrieving the data
		hdul = fits.open(os.path.join(file_path, img))
		img, ra, dec = data_from_fits_files(hdul)

		#Performing the flipping
		start = time.time()

		if ANGLE_ROTATE == True:
			new_image = imrotate(img, angle, interp = 'bilinear')

		else:
			if VERTICAL_FLIP == True:
				new_image = np.flipud(img)
			else:
				new_image = np.fliplr(img)


		write_fits_from_data(new_image, file_path, base_file_name + f"{angle}_{i + 1}", ra, dec)

		print(f"Created rotated fits files for image {i} in time: {time.time() - start}")

################################# USAGE #################################

angle = [1, -1, 2, -2, 3, -3, 90, 270]

for i in angle:
	fits_image_augment(FILE_PATH, BASE_FILE_NAME, i, True, True)
