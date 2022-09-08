# -*- coding: utf-8 -*-
"""
Based on: https://github.com/jmather625/predicting-poverty-replication/blob/master/scripts/download_images.ipynb
"""

#### To be run after Sample.py ####

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import time
import socket
import random

BASE_DIR = "H:\\GDP\\GDP_prediction_minimal_implementation\\" #I should really rename this as root_dir.
ACCESS_TOKEN_DIR = "H:\\GDP\\GDP_prediction_minimal_implementation\\google_api_key.txt" #This bad boy is critical for accessing Google's Static Map API which MUST be configured properly first.
sys.path.append(BASE_DIR)

from utils.google_downloader import GoogleDownloader
from utils.CountryIterator import CountryIterator

clipped_images_directory = r"C:/Data/Nightlights/2020/CountryClips/"
county_list_file = os.path.join(clipped_images_directory,"counties.csv")
counties = CountryIterator.get_county_codes(county_list_file)

sample_list_folder = os.path.join(BASE_DIR,"subnational_samples/")
coordinate_write_file = "{}_{}_sample.txt"

save_path = "C:\Data\Daytime_Imagery\subnational_study"
download_list_pattern = "{}_{}_locs.csv"

access = open(ACCESS_TOKEN_DIR, 'r').readlines()[0]
gd = GoogleDownloader(access)


def download_images(df):
    """
    Download images using a pandas DataFrame that has "image_lat", "image_lon", "image_name", "country" as columns
    
    Saves images to the corresponding country's images folder
    """
    num_retries = 20
    wait_time = 0.1 # seconds
    no_internet_wait_time = 300 #seconds

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        lat = r.lats
        lon = r.lons
        name = r.im_name
        name_a = name + 'a'
        image_save_path = os.path.join(current_save_dir, name)
        fails = 0
        if os.path.isfile(image_save_path):
            continue
        try:
            im = gd.download(lat, lon, 14, 250, 250, True)
            if (type(im) == str and im == 'RETRY') or im is None:
                resolved = False
                for _ in range(num_retries):
                    #if not internet_on():
                    if fails > 10:
                        time.sleep(no_internet_wait_time)
                        print("Uh oh, internet's down. Waiting "+str(no_internet_wait_time)+" seconds.")
                        fails = 0
                    else:
                        time.sleep(np.random.random()*2)#wait_time)
                    im = gd.download(lat, lon, 14, 250, 250, True)
                    if (type(im) == str and im == 'RETRY') or im is None:
                        continue
                    else:
                        plt.imsave(image_save_path, im)
                        resolved = True
                        break
                if not resolved:
                    print(f'Could not download {lat}, {lon} despite several retries and waiting')
                    fails +=1 
                    continue
                else:
                    pass
            else:
                # no issues, save according to naming convention
                plt.imsave(image_save_path, im)
        except Exception as e:
            logging.error(f"Error-could not download {lat}, {lon}", exc_info=True)
            continue
        
def download_image(name, lat, lon):
    num_retries = 20
    wait_time = np.random.random() # seconds
    image_save_path = os.path.join(current_save_dir, name)
    try:
        im = gd.download(lat, lon, 14, 400, 400, True)
        if (type(im) == str and im == 'RETRY') or im is None:
            resolved = False
            for _ in range(num_retries):
                time.sleep(np.random.random()*2)
                im = gd.download(lat, lon, 14, 400, 400, True)
                if (type(im) == str and im == 'RETRY') or im is None:
                    continue
                else:
                    plt.imsave(image_save_path, im)
                    resolved = True
                    break
            if not resolved:
                print(f'Could not download {lat}, {lon} despite several retries and waiting')
                return None
            else:
                return None
        else:
            # no issues, save according to naming convention
            plt.imsave(image_save_path, im)
            
    except Exception as e:
        logging.error(f"Error-could not download {lat}, {lon}", exc_info=True)
        return None
            
#Going to download in batches so this will be necessary.
download_indices = range(0,400)

i=-1
for _, r in counties.iterrows():
    i += 1
    if i not in download_indices:
        continue
    print("Starting " + r.state + " " + r.county)
    coords = np.genfromtxt(sample_list_folder + coordinate_write_file.format(r.state,r.county), delimiter=",")
    sample_size = coords.shape[1]

    locs = pd.DataFrame( #Dataframe of sample point locations.
        {   
            'im_name': [r.state + "_" + r.county + "_" + str(i) + ".png" for i in range(0,sample_size)],
            'lats': coords[0],
            'lons': coords[1],
            'state': [r.state for i in range(0,sample_size)],
            'county': [r.county for i in range(0,sample_size)]
        }
    )

    current_save_dir = os.path.join(save_path, r.state+"_"+r.county)
    os.makedirs(current_save_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(current_save_dir, download_list_pattern.format(r.state,r.county))):
        print("Skipping " + download_list_pattern.format(r.state,r.county))
        continue
    
    locs.to_csv(os.path.join(current_save_dir, download_list_pattern.format(r.state,r.county)), sep=",")

    download_images(locs)
    
#Check to see if all the files are there, else retry
download_indices = range(0,400)

i=-1
for _, r in tqdm(counties.iterrows()):
    i += 1
    if i not in download_indices:
        continue
    current_save_dir = os.path.join(save_path, r.state+"_"+r.county)
    os.makedirs(current_save_dir, exist_ok=True)
    locs = pd.read_csv(os.path.join(current_save_dir, download_list_pattern.format(r.state,r.county)), sep=',')
    for i2, r2 in locs.iterrows():
        fname = os.path.join(current_save_dir, r2.im_name)
        if not os.path.exists(fname):
            print(r2.im_name + ' does not exist :(')
            download_image(r2.im_name, r2.lats, r2.lons)