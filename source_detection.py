################################# IMPORTS #################################
import pandas, time, astropy, sep, skimage, os, cv2
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma, mad_std
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.wcs import WCS
from astropy.table import Table
from photutils import source_properties, deblend_sources, detect_sources
from astropy.coordinates import Angle, SkyCoord

################################# CLASS AND FUNCTIONS #################################
class SourceDetection:

	def background_reduction(imgPath):
		'''
		Performing a background reduction on the fits images
		Arguments:
			imgFile: Path to the fits file that is requiring background reduction

		'''

        print('Opening fits file and subtracting background')
        start = time.time()

        #opening the header of the major fits file
        hdu1 = fits.open(imgPath) 

        #obtaining a signal to noise ratio of the image, can be used instead of calculating a background for detection
        data = hdu1[1].data
        hdr = hdu1[1].header
        var = hdu1[3].data
        sn_map = data/np.sqrt(var)

        #remove background from image - code from sep tutorial. - https://sep.readthedocs.io/en/v1.0.x/tutorial.html
        bkg = sep.Background(data)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        
        print(f"Performed background reduction in {time.time() - start}")

        # Returning the background subtracted image 
        subtracted_image = data - bkg
        return subtracted_image

    def find_sources(data_sub, x_size = 5, y_size = 5, npixels = 10, connectivity = 8):
    	'''
		Using photutils to detect the sources within the full fits files

		Arguments:
			data_sub: Background subtracted fits file

		Optional Arguments:
			x_size: x extent of the kernel which slides over the image to detect the sources 
					-- defaults to 5 pixels
			y_size: y extent of the kernel which slides over the image to detect the sources 
					-- defaults to 5 pixels
			n_pixels: number of connected pixels that are greater than the threshold to count a source 
					-- defaults to 10 pixels
			connectivity: The type of pixel connectivity used in determining how pixels are grouped into a detected source.
					-- defaults to 8 pixels which touch along their edges or corners.

    	'''

    	print('Finding sources using photutils')
    	start = time.time()

    	median = np.median(data_sub)
        std = mad_std(data_sub)

        threshold = bkg + (5.0 * bkg_rms)
        sigma = 5.0 * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size = x_size, y_size = y_size) # Kernel defaults to 8*stddev
        kernel.normalize()

        segmented_image = detect_sources(data_sub, threshold, npixels = npixels, filter_kernel = kernel, connectivity = connectivity)
        segmented_image_deblend = deblend_sources(data_sub, segmented_image, npixels = npixels, filter_kernel = kernel, connectivity = connectivity)

        cat = source_properties(data_sub, segmented_image_deblend)

        # Getting values of the individual stars to place into a table
        x_pos = cat.xcentroid.value
        y_pos = cat.ycentroid.value
        area = cat.area.value
        max_pixel_val = cat.max_value
        ids = cat.id

        return ids, x_pos, y_pos, area, max_pixel_val

    def compare_to_catalogue(hdr, x_pos, y_pos, catalogue_path, ids, area, max_pixel_val, csv_name):
    	'''
		Creates a csv table of the details of the detected sources. Compares to a chosen catalogue to determine what
		type of astronomical object is present.

		Arguments:
			hdr: Header of the major fits file
			x_pos: List of the x positions of the objects that have been detected in the image
			y_pos: List of the y positions of the objects that have been detected in the image  
			catalogue_path: A csv file of an astronomical survey that contains RA and DEC of sources. Must be sources that
							are within the same area of sky in order to correctly determine what object was detected.
							For the project the Galaxy Zoo 2 catalogue was used
			ids: IDs of the sources determined by source detection
			area: Area of the detected source determined by source detection
			max_pixel_val: Maximum pixel value of the source determined by source detection
			csv_name: Name of the file path that the table will be saved to

    	'''

    	#get world coordinate data from header of image.
        wcs = WCS(hdr)

        #get RA and DEC at pixel positions for each source using wcs
        ra, dec = wcs.all_pix2world([x_pos],[y_pos],1) 
        ra = ra[0]
        ra = Angle(ra, u.deg)
        ra_goto = ra.hour
        dec_goto = dec[0]

        galzoo = pandas.read_csv(catalogue_csv)
        ra = galzoo.ra
        dec = galzoo.dec
        dec = Angle(dec, u.degree)
        ra = Angle(ra, u.degree)
        dec_galzoo = dec.deg
        ra_galzoo = ra.deg

        #get the type of galaxy from gal zoo
        galaxy_type = galzoo.gz2_class
        galaxy_type_simple = []
        for i in range(len(galaxy_type)):
            test=''.join([x[0] for x in galaxy_type[i].split()])
            galaxy_type_simple.append(test)

        ra_goto = Angle(ra_goto,u.hour)
        ra_goto = ra_goto.deg
        dec_goto = Angle(dec_goto,u.deg)
        dec_goto = dec_goto.deg

        c = SkyCoord(ra=ra_goto*u.degree, dec=dec_goto*u.degree)
        catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        idx,d2d,d3d = c.match_to_catalog_sky(catalog)
        d2d_degree = d2d.degree
    
    	# Creating a table that compares the detected stars with the catalogue stars
    	
    	# Keeping track of number of galaxies and stars in the image
    	numb_stars = 0 
        numb_spiral = 0
        numb_ellip = 0

        # Empty arrays for each detected source.
        actual_galaxies = [] 
        actual_galaxy_index = []
        galaxy_id = []
        imaged_galaxy_types = []
        for i in range(len(d2d)):

        	#If distance between source and galaxy in galzoo is <1 arc_sec.
            if d2d_degree[i] <= 0.000555556: 
                actual_galaxies.append(True)
                actual_galaxy_index.append(i)
                galaxy_id.append(idx[i])
                imaged_galaxy_types.append(galaxy_type_simple[idx[i]])
                if galaxy_type_simple[idx[i]]=='S': numb_spiral += 1
                if galaxy_type_simple[idx[i]]=='E': numb_ellip += 1 
            else:
                actual_galaxies.append(False)
                imaged_galaxy_types.append('NA')
                numb_stars += 1

        table = Table([ids, x_pos, y_pos, area, max_pixel_val, ra_goto ,dec_goto ,actual_galaxies ,imaged_galaxy_types], 
            names = ('id', 'x_pixel', 'y_pixel', 'area', 'max_pixel_val', 'RA' ,'Dec' ,'actual_galaxy' ,'galaxy_type'))
        table.write(csv_name, format='csv',overwrite=True)

        print(f'CSV file created')
        print(f'Total number of stars: {numb_stars}')
        print(f'Total number of spiral galaxies: {numb_spiral}')
        print(f'Total number of elliptical galaxies: {numb_ellip}') 

    def cutout_sources(img, csv_table, cutout_folder, file_number, IMG_SIZE = 30):
    	'''
    	Creates individual cutouts of the sources detected of a selected size, and places
    	the resulting fits files in folders ready to have analysis performed on them ready
    	for feeding to the Convolutional Neural Network

    	Arguments:
    		img: The background subtracted image with detected sources in.
    		csv_table: Table of detected sources with full pixel information
    		cutout_folder: Base folder that the cutout images will be saved in.
    		file_number: number of the base file that the cutouts are taken from - primarily used to iterate through a list

    	Optional Arguments:
    		IMG_SIZE: x and y dimensions of the cutout images in pixels
    				--- defaults to 30
    	'''

    	# Extracting the data from the created table
    	data = pandas.read_csv(csv_table)
    	ra = data.RA
        dec = data.Dec
        x_pos = data.x_pixel
        y_pos = data.y_pixel
        galaxy_type = data.galaxy_type
        area = data.area
        max_pix_val = data.max_pixel_val

        # Creating the cutouts
        start = time.time()
        if 'E' or 'S' in galaxy_type:
            print('Cutting out individual images from the background subtracted image, saving their RA and DEC')
            for i in range(len(x_pos)):
            	# If sources are within IMG_SIZE / 2 pixels of the edge ignore, so cutouts are centred
                if  (IMG_SIZE / 2) < x_pos[i] < img.shape[1] - (IMG_SIZE / 2) and (IMG_SIZE / 2) < y_pos[i] < img.shape[0] - (IMG_SIZE / 2): 
                    cutout_img = astropy.nddata.Cutout2D(img_sub, position = (x_pos[i], y_pos[i]), size = (IMG_SIZE, IMG_SIZE)).data
                    
                    # Get data (in fits form) from the cutout
                    hdu = fits.PrimaryHDU(cutout_img) 
                    hdu_header = hdu1[0].header

                    # Set header of each cutout to have RA and DEC
                    hdu.header.set('RA',ra[i])
                    hdu.header.set('DEC',dec[i])

                    # Saving the fits files to a folder in a specific format

                    ################################ GALAXIES ################################
                    if galaxy_type[i] == 'E':
                        hdu.writeto(f"{cutout_folder}/elliptical_galaxies_original/elliptical_galaxy{file_number}_{i + 1}.fits", overwrite = True)
                        hdu.writeto(f"{cutout_folder}/all_galaxies_original/galaxy{file_number}_{i + 1}.fits",overwrite = True) #to create a file with all galaxies in it...    
                    if galaxy_type[i] == 'S':
                        hdu.writeto(f"{cutout_folder}/spiral_galaxies_original/spiral_galaxy{file_number}_{i + 1}.fits", overwrite = True)
                        hdu.writeto(f"{cutout_folder}/all_galaxies_original/galaxy{file_number}_{i + 1}.fits", overwrite = True)

                    # Attemps to remove any anomalous objects (artefacts) from the pool of images used for machine learning
                    if area[i] < 75 or np.sqrt(((x_pos[i]-x_pos[i - 1])**2 + (y_pos[i] - y_pos[i - 1])**2)) < np.sqrt((IMG_SIZE / 2)**2 + (IMG_SIZE/2)**2):
                        hdu.writeto(f"{cutout_folder}/artefacts/source{file_number}_{i + 1}.fits", overwrite=True) 
                    else:
                    	################################# STARS ################################
                        hdu.writeto(f"{cutout_folder}/stars/star{number}_{i + 1}.fits",overwrite = True)                   
        else: print('No galaxy present in the image')

        print(f"Finished cutting out images, time = {time.time() - start}")


################################ USAGE ################################
CATALOGUE_PATH = 'gz2_hart16.csv'

def prepare_data(imgPath, csv_table, cutout_folder, NUMBER_OF_BASE_IMAGES = 13, BASE_FILE_PATH = '142/', BASE_DATA_PATH = '142_data/'):
	for i in range(NUMBER_OF_BASE_IMAGES):
		for j in range(NUMBER_OF_BASE_IMAGES):
			bkg_reduced_image = SourceDetection.background_reduction(BASE_FILE_PATH + f"{i}_{j}.fits")
			ids, x_pos, y_pos, area, max_pixel_val = SourceDetection.find_sources(bkg_reduced_image)

			# Opening up the header of the background reduced image      
        	hdu1 = fits.open(bkg_reduced_image) 
        	hdr = hdu1[1].header

			SourceDetection.compare_to_catalogue(hdr, x_pos, y_pos, CATALOGUE_PATH, ids, area, max_pixel_val, BASE_DATA_PATH + f"/source{i}_{j}")

			SourceDetection.cutout_sources(bkg_reduced_image, csv_table, cutout_folder, (i * NUMBER_OF_BASE_IMAGES) + j)

