import pandas, pdb, time, warnings, astropy
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy import optimize
from pylab import meshgrid
from astropy.stats import sigma_clipped_stats, sigma_clip, gaussian_fwhm_to_sigma, mad_std
from astropy.io import fits
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table, Column
from astropy.modeling import models, fitting
from photutils import find_peaks, CircularAperture, source_properties, deblend_sources, detect_sources
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky
import sep
import skimage
import os
import cv2


def prepare_data():

    def create_table(imgFile,csv_name):
        print('Opening fits file and subtracting background')
        start = time.time()
        hdu1 = fits.open(imgFile) #opening the header of the major fits file
        hdr = hdu1[1].header
        #obtaining a signal to noise ratio of the image, can be used instead of calculating a background for detection
        data = hdu1[1].data
        var = hdu1[3].data
        sn_map = data/np.sqrt(var)
        #remove background from image - code from sep tutorial. - https://sep.readthedocs.io/en/v1.0.x/tutorial.html
        bkg = sep.Background(data)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        data_sub = data - bkg #background subtracted data.

        print('Finding sources using photutils, time =', time.time()-start)
        median = np.median(data_sub)
        std = mad_std(data_sub)
        threshold = bkg + (5.0 * bkg_rms)
        sigma = 5.0 * gaussian_fwhm_to_sigma
        kernel = Gaussian2DKernel(sigma, x_size = 5, y_size = 5) #kernel defaults to 8*stddev
        kernel.normalize()
        segmented_image = detect_sources(data_sub, threshold, npixels = 10, filter_kernel = kernel, connectivity = 8)
        print('Deblending sources, time =', time.time()-start)
        segmented_image_deblend = deblend_sources(data_sub, segmented_image, npixels = 10, filter_kernel = kernel, connectivity = 8)

        cat = source_properties(data_sub, segmented_image_deblend)
        x_pos = cat.xcentroid.value
        y_pos = cat.ycentroid.value
        area = cat.area.value
        max_pixel_val = cat.max_value
        ids = cat.id

        print('Comparing sources to galaxy zoo via their RA and DEC')
        #get world coordinate data from header of image.
        wcs = WCS(hdr)
        ra,dec = wcs.all_pix2world([x_pos],[y_pos],1) #get RA and DEC at pixel positions for each source using wcs
        ra = ra[0]
        ra = Angle(ra, u.deg)
        ra_goto = ra.hour
        dec_goto = dec[0]

        galzoo = pandas.read_csv('gz2_hart16.csv')
        ra = galzoo.ra
        dec = galzoo.dec
        dec = Angle(dec, u.degree)
        ra = Angle(ra, u.degree)
        dec = dec.deg
        ra = ra.deg

        galaxy_type = galzoo.gz2_class#get the type of galaxy from gal zoo
        galaxy_type_simple = []
        for i in range(len(galaxy_type)):
            test=''.join([x[0] for x in galaxy_type[i].split()])
            galaxy_type_simple.append(test)

        ra_goto = Angle(ra_goto,u.hour)
        ra_goto = ra_goto.deg
        dec_goto = Angle(dec_goto,u.deg)
        dec_goto = dec_goto.deg

        from astropy.coordinates import SkyCoord
        c = SkyCoord(ra=ra_goto*u.degree, dec=dec_goto*u.degree)
        catalog = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
        idx,d2d,d3d = c.match_to_catalog_sky(catalog)
        d2d_degree = d2d.degree


        numb_stars = 0 #keep track of number of galaxies and stars in the image, just to know.
        numb_spiral = 0
        numb_ellip = 0

        actual_galaxies = [] #empty array to and true and false for each detected source.
        actual_galaxy_index = []
        galaxy_id = []
        imaged_galaxy_types = []
        for i in range(len(d2d)):
            if d2d_degree[i] <= 0.000555556: #If distance between source and galaxy in galzoo is <1 arc_sec.
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
        print('Created a csv file, according to our source detection there are {0} stars, and according to galaxy zoo2 database there are {1} spiral and {2} elliptical galaxies within the image'.format(numb_stars,numb_spiral,numb_ellip)) 

    def cutout_sources(imgFile,data,cutout_folder,number):
        hdu1 = fits.open(imgFile) #opening the header of the major fits file
        img = hdu1[1].data
        hdr = hdu1[1].header
        bkg = sep.Background(img)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()
        img_sub = img - bkg #background subtracted data.

        data = pandas.read_csv(data)
        ra = data.RA
        dec = data.Dec
        x_pos = data.x_pixel
        y_pos = data.y_pixel
        galaxy_type = data.galaxy_type
        area = data.area
        max_pix_val = data.max_pixel_val
        IMG_SIZE = 30
        start = time.time()
        if 'E' or 'S' in galaxy_type:
            print('cutting out individual images from the background subtracted image, saving their RA and DEC')
            for i in range(len(x_pos)): #loop through all the sources.
                if  (IMG_SIZE/2) < x_pos[i] < img.shape[1]-(IMG_SIZE/2) and (IMG_SIZE/2) < y_pos[i] < img.shape[0]-(IMG_SIZE/2): #if sources are within 30 pixels of the edge ignore, so cutouts are centred
                    cutout_img = astropy.nddata.Cutout2D(img_sub,position=(x_pos[i],y_pos[i]),size=(IMG_SIZE,IMG_SIZE)).data #cut out 50x50 image located at x position and y posit
                    hdu = fits.PrimaryHDU(cutout_img) #get data (in fits form) from the cutout
                    hdu_header = hdu1[0].header #get header (empty atm)
                    hdu.header.set('RA',ra[i]) #set header of each cutout to have RA and DEC
                    hdu.header.set('DEC',dec[i])
                    if galaxy_type[i] == 'E':
                        hdu.writeto('{0}/elliptical_galaxies_original/elliptical_galaxy{1}_{2}.fits'.format(cutout_folder,number,i+1),overwrite = True)
                        hdu.writeto('{0}/all_galaxies_original/galaxy{1}_{2}.fits'.format(cutout_folder,number,i+1),overwrite = True) #to create a file with all galaxies in it...    
                    if galaxy_type[i] == 'S':
                        hdu.writeto('{0}/spiral_galaxies_original/spiral_galaxy{1}_{2}.fits'.format(cutout_folder,number,i+1),overwrite = True)
                        hdu.writeto('{0}/all_galaxies_original/galaxy{1}_{2}.fits'.format(cutout_folder,number,i+1),overwrite = True)
                    if area[i] < 75 or np.sqrt(((x_pos[i]-x_pos[i-1])**2 + (y_pos[i]-y_pos[i-1])**2)) < np.sqrt((IMG_SIZE/2)**2 + (IMG_SIZE/2)**2):
                        hdu.writeto('{0}/artefacts/source{1}_{2}.fits'.format(cutout_folder,number,i+1), overwrite=True) 
                    else:
                        hdu.writeto('{0}/stars/star{1}_{2}.fits'.format(cutout_folder,number,i+1),overwrite = True) #write out the cuts to the same file the code is saved to, will overwrite files with the same name.                    
        else: print('No galaxy present in the image')
        print('Finished cutting out images, time = ', time.time()-start)

    for x in range(13):
        for i in range(13):
            try:
                create_table('142/{0},{1}.fits'.format(x,i),'142_data/source_{0},{1}.csv'.format(x,i))
                print('Currently finding sources and creating csv files for image',x,i)
                cutout_sources('142/{0},{1}.fits'.format(x,i),'142_data/source_{0},{1}.csv'.format(x,i),'142_cutouts',(x*13)+i)
                print('Currently creating cutouts for image',x,i)
            except Exception as e:
                pass 

prepare_data()

