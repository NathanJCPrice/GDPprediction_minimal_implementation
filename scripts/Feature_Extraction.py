# -*- coding: utf-8 -*-
"""
This file runs the images through the feature extractor.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms

root_dir=''
os.chdir(root_dir)

from utils.visualization_utils2 import preprocess_image, convert_to_grayscale

images_main_path = "daytime_imagery\\"
model_path = "trained_model.pt" #This should be the file path of J Mather's trained model. See the readme file.
clipped_images_directory = ""
country_list_file = "country_list.txt"
download_list_pattern = "{}_locs.csv"
features_file_pattern = "{}_features.csv"

input_size = 224 #Image width/height in pixels.
n_feats = 4096 #Dimension of the representation space before dimensionality reduction.
n_samps = 1000 #Number of sample points per country.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.classifier = model.classifier[:4] #Bye bye last 2 layers.

# Import countries list.
countries = []
with open(clipped_images_directory + country_list_file,"r") as country_list:
    for line in country_list:
        country = line.replace("\n","")
        countries.append(country)
        
for country in countries:
    country_image_folder = os.path.join(images_main_path,country)
    locs = pd.read_csv(os.path.join(country_image_folder, download_list_pattern.format(country)), sep=',')

    country_feats = np.empty((n_samps,n_feats))

    for i, r in tqdm(locs.iterrows(), total=locs.shape[0]):
        im_path = os.path.join(country_image_folder, r.im_name)
        im = Image.open(im_path).convert('RGB')
        lb = int(200-input_size/2)
        ub = int(200+input_size/2)
        proc_image = preprocess_image(im,resize_im=False)[:,:,lb:ub,lb:ub]

        with torch.no_grad():
            country_feats[i] = model(proc_image)
        
    np.savetxt(os.path.join(images_main_path,features_file_pattern.format(country)), country_feats, delimiter=',')
    print(country + " complete")
    
        
        