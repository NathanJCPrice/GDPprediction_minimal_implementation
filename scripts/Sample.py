# -*- coding: utf-8 -*-
"""
This class handles the importance sampling.
"""

import os
import sys

root_dir = '' #Set to working directory.
os.chdir(root_dir) 
sys.path.append(root_dir)

from utils.geosampler import GeoSampler
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

#Unfortunately, this seed doesn't recreate the sample from the paper.
#The original study area was not global and was only made global later.
#Consequently, the sampling was done in a few runs of this file.
np.random.seed(3660)

clipped_images_directory = root_dir+"country_nightlight_rasters\\"
filename_pattern = "Joined_VIIRS_{}.tif"
country_list_file = "country_list.txt"

save_folder = root_dir+"samples\\" #Where the samples should be saved.
geobounds_pattern = "{}_geobounds.tif" #Filepath pattern for a file that gives the country borders / bounds.
pps_pattern = "{}_pps.tif" #Filepath pattern for the file that gives the pps sample's first stage.
coordinate_write_file = "{}_sample.txt" #Gives the coordinate list.

n = 1000 #Number of points to sample per country.

countries = []

with open(clipped_images_directory + country_list_file,"r") as country_list:
    for line in country_list:
        countries.append(line.replace("\n",""))

count = 0
for country in countries:
    country_path = clipped_images_directory + filename_pattern.format(country)
    if os.path.isfile(save_folder + coordinate_write_file.format(country)) or country not in ("Kiribati"): #Kiribati was excluded from analysis. See the paper.
        print("Skipping " + country)
        continue
        
    print("Starting " + country)

    sampler = GeoSampler(country_path) #The GeoSampler class handles all the sampling logic.
    sampler.initialise_selection_probabilities(0.01, True) #Epsilon=0.01. See paper for details about this parameter.
    bounds_image = sampler.visualise_geobounds()
    bounds_image.save(save_folder + geobounds_pattern.format(country),"TIFF") #I can't remember what the point of this file is. Might just be for debugging?

    sampler.PPS_sample(n, progress_bar=True)
    sample_image = sampler.visualise_sample()
    sample_image.save(save_folder + pps_pattern.format(country),"TIFF") #I think this file is also for debugging.

    sampler.get_sample_points()
    np.savetxt(save_folder + coordinate_write_file.format(country), sampler.sample_points, delimiter=',')
        
    print(country + " complete")


