# -*- coding: utf-8 -*-
"""
This class handles the importance sampling.
"""

import os
import sys

root_dir = 'H:\\GDP\\GDP_prediction_minimal_implementation\\' #Set to working directory.
os.chdir(root_dir) 
sys.path.append(root_dir)

from utils.geosampler import GeoSampler
from utils.CountryIterator import CountryIterator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(7775)

clipped_images_directory = r"C:/Data/Nightlights/2020/CountryClips/"
filename_pattern = "Joined_VIIRS_{}_{}.tif"
county_list_file = os.path.join(clipped_images_directory,"counties.csv")

save_folder = os.path.join(root_dir,"subnational_samples\\") #Where the samples should be saved.
geobounds_pattern = "{}_{}_geobounds.tif" #Filepath pattern for a file that gives the country borders / bounds.
pps_pattern = "{}_{}_pps.tif" #Filepath pattern for the file that gives the pps sample's first stage.
coordinate_write_file = "{}_{}_sample.txt" #Gives the coordinate list.

n = 250 #Number of points to sample per country.

county_codes = CountryIterator.get_county_codes(county_list_file)

count = 0
for _, r in county_codes.iterrows():
    county = r.county
    state = r.state
    county_path = clipped_images_directory + filename_pattern.format(state,county)
    if os.path.isfile(save_folder + coordinate_write_file.format(state,county)):
        print("Skipping " + county)
        continue
        
    print("Starting " + county)

    sampler = GeoSampler(county_path) #The GeoSampler class handles all the sampling logic.
    sampler.initialise_selection_probabilities(0.01, True) #Epsilon=0.01. See paper for details about this parameter.
    bounds_image = sampler.visualise_geobounds()
    bounds_image.save(save_folder + geobounds_pattern.format(state,county),"TIFF") #I can't remember what the point of this file is. Might just be for debugging?

    sampler.PPS_sample(n, progress_bar=True)
    sample_image = sampler.visualise_sample()
    sample_image.save(save_folder + pps_pattern.format(state,county),"TIFF") #I think this file is also for debugging.

    sampler.get_sample_points()
    np.savetxt(save_folder + coordinate_write_file.format(state,county), sampler.sample_points, delimiter=',')
        
    print(county + " complete")


