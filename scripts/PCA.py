# -*- coding: utf-8 -*-
"""
This file computes the PCA and then performs it on all the representations.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from PIL import Image
from utils.geosampler import GeoSampler
from utils.CountryIterator import CountryIterator

root_dir = 'H:\\GDP\\GDP_prediction_minimal_implementation\\'
os.chdir(root_dir)

images_main_path = "C:\\Data\\Daytime_Imagery\\Paper1\\"
clipped_images_directory = "H:\\GDP\\"
clipped_pattern = "Joined_VIIRS_{}.tif"
country_list_file = "country_list.txt"
download_list_pattern = "{}_locs.csv"
features_file_pattern = "{}_features.csv"
reduced_features_file_pattern = "{}_reduced.csv" #The file pattern to be used for the reduced dimension representations.
means_then_vars_path = root_dir+"means_then_vars2.csv"
final_dataset_path = root_dir+"image_features2.csv"
long_locs_path = root_dir+"full_sample2.csv" #This is the file which will contain all the modelling info. I should rename this.
download_list_pattern = "{}_locs.csv"


n_feats = 4096 #Number of dimensions of the features before dimension reduction.
n_samps = 1000 #Number of sample points per country.
dimensions = 12 #The number of dimensions to reduce to.

countries = []
with open(clipped_images_directory + country_list_file,"r") as country_list:
    for line in country_list:
        country = line.replace("\n","")
        if country not in ("","Kiribati"):
            countries.append(country)
        
sklearn_pca = IncrementalPCA(n_components=dimensions)

#The only point of this PCA is to get the means and variances. I am lazy.        
for country in tqdm(countries):
    features_path = os.path.join(images_main_path,features_file_pattern.format(country))
    features = np.genfromtxt(features_path, delimiter=",")
    sklearn_pca.partial_fit(features)
    
print("PCA complete. Starting normalised PCA.")
    
means_then_vars = np.empty((2,n_feats))
means_then_vars[0] = sklearn_pca.mean_
means_then_vars[1] = sklearn_pca.var_
np.savetxt(means_then_vars_path, means_then_vars, delimiter=',')

means = means_then_vars[0]
std = np.sqrt(means_then_vars[1])

def normalise(data):
    return (data-means)/std

#We normalise all dimensions before the PCA.
normalised_pca = IncrementalPCA(n_components=dimensions)

#This is the real PCA.        
for country in tqdm(countries):
    features_path = os.path.join(images_main_path,features_file_pattern.format(country))
    features = np.genfromtxt(features_path, delimiter=",")
    features = normalise(features)
    normalised_pca.partial_fit(features)
    
print("Normalised PCA complete. Starting transforms.")
    
for country in tqdm(countries):
    features_path = os.path.join(images_main_path,features_file_pattern.format(country))
    features = np.genfromtxt(features_path, delimiter=",")
    features = normalise(features)
    reduced_features = normalised_pca.transform(features)
    np.savetxt(os.path.join(images_main_path,reduced_features_file_pattern.format(country)), reduced_features, delimiter=',')
    
df = None
for country in tqdm(countries):
    country_image_folder = os.path.join(images_main_path,country)
    locs = pd.read_csv(os.path.join(country_image_folder, download_list_pattern.format(country)), sep=',')
    reduced_features_path = os.path.join(images_main_path,reduced_features_file_pattern.format(country))
    reduced_features = np.genfromtxt(reduced_features_path, delimiter=",")
    temp_dataset = pd.DataFrame(data=reduced_features, index=np.array(range(0,n_samps)), columns=["N"+str(i) for i in range(0,dimensions)])
    big_locs = locs.join(temp_dataset)
    if df is None:
        df = big_locs
    else:
        df = df.append(big_locs)
        
df.to_csv(final_dataset_path, sep=",")

#This adds weights and radiance values to each observation.
def weight_and_add_values_to_locs():
    for country, data, fpath in CountryIterator(countries, images_main_path, download_list_pattern, library="pandas", output_filepath=True, individual_country_folders=True):
        sampler_initialised = False
        n = data.shape[0]
        country_path = clipped_images_directory + clipped_pattern.format(country)
        if not "weight" in data.columns:
            sampler = GeoSampler(country_path)
            sampler_initialised = True
            
            data.insert(data.shape[1], "weight", 0)
            sampler.initialise_selection_probabilities(0.01, True)
            sampler.set_weights(n)
            weights = sampler.get_weight_at_proj_pos(data)
            data['weight'] = weights
        else:
            print("Weight column already exists in "+ download_list_pattern.format(country))
        if not "radiance" in data.columns:
            if not sampler_initialised:
                sampler = GeoSampler(country_path)
                sampler_initialised = True
            data.insert(data.shape[1], "radiance", 0)
            data['radiance'] = sampler.get_value_at_proj_pos(data)
        else:
            print("Radiance column already exists in "+ download_list_pattern.format(country))
        data.to_csv(fpath,sep=',')
        
weight_and_add_values_to_locs()
        
#This brings all the sample points into one data frame.
def make_long_locs(output_path):
    first = True
    for country in tqdm(countries):
        country_image_folder = os.path.join(images_main_path,country)
        locs = pd.read_csv(os.path.join(country_image_folder, download_list_pattern.format(country)), sep=',')
        if first:
            longcs = locs
            first = False
        else:
            longcs = longcs.append(locs)
    return longcs

long_locs = make_long_locs(long_locs_path)

#This function adds the representation columns onto the sample dataframe.
def select_columns(folder, pattern, feature_indices, long_locs_path=None):
    feature_indices_sorted = feature_indices.copy()
    feature_indices_sorted.sort()
    try: 
        long_locs
    except NameError:
        raise Exception("Run make_long_locs first.")
    features = ["N"+str(i) for i in feature_indices_sorted]
    first = True
    for country in tqdm(countries):
        file_path = folder + pattern.format(country)
        country_image_folder = os.path.join(images_main_path,country)
        cols = np.genfromtxt(file_path, delimiter=",")[:,feature_indices]
        if first:
            big_dataset = cols
            first = False
        else:
            big_dataset = np.concatenate([big_dataset,cols],axis=0)
    locs_copy = long_locs.copy()
    for i in range(0,len(feature_indices_sorted)): 
        locs_copy.insert(locs_copy.shape[1],features[i],big_dataset[:,i])
    return locs_copy

long_locs = select_columns(images_main_path, reduced_features_file_pattern, np.array(range(dimensions)))
        
long_locs.to_csv(long_locs_path)
