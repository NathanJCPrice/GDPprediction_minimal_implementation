# -*- coding: utf-8 -*-
"""
This file runs the images through the feature extractor.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image

root_dir='H:\\GDP\\GDP_prediction_minimal_implementation\\'
os.chdir(root_dir)

from utils.visualization_utils2 import preprocess_image
from utils.CountryIterator import CountryIterator

images_main_path = "C:/Data/Daytime_Imagery/subnational_study"
model_path = "C:\\Data\\Sustain_model\\trained_model.pt" #This should be the file path of J Mather's trained model. See the readme file.
county_list_file = os.path.join(r"C:\Data\Nightlights\2020\CountryClips","counties.csv")
download_list_pattern = "{}_{}_locs.csv"
features_file_pattern = "{}_{}_features.csv"

input_size = 224 #Image width/height in pixels.
n_feats = 4096 #Dimension of the representation space before dimensionality reduction.
n_samps = 250 #Number of sample points per country.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.classifier = model.classifier[:4] #Bye bye last 2 layers.

# Import counties list.
counties = CountryIterator.get_county_codes(county_list_file)

i=-1
for _, r in counties.iterrows():
    i += 1
    if i not in range(0,400):
        continue
    county = r.county
    state = r.state
    county_image_folder = os.path.join(images_main_path,state+"_"+county)
    locs = pd.read_csv(os.path.join(county_image_folder, download_list_pattern.format(state,county)), sep=',')

    county_feats = np.empty((n_samps,n_feats))

    for i, r in tqdm(locs.iterrows(), total=locs.shape[0]):
        im_path = os.path.join(county_image_folder, r.im_name)
        im = Image.open(im_path).convert('RGB')
        lb = int(200-input_size/2)
        ub = int(200+input_size/2)
        proc_image = preprocess_image(im,resize_im=False)[:,:,lb:ub,lb:ub]

        with torch.no_grad():
            county_feats[i] = model(proc_image)
        
    np.savetxt(os.path.join(images_main_path,features_file_pattern.format(state,county)), county_feats, delimiter=',')
    print(county + " complete")
    
        
        