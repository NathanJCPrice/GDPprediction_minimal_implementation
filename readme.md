# Global GDP Prediction Python implementation

## Intro
This is the code for the paper *'Global GDP Prediction With Nightlights and Transfer Learning'* by Nathan Price and Peter Atkinson.

Let me start by apologising for the hacked together code. I mean to writer a cleaner implementation when I have time, but the code works and it seems better to open source it than to keep it hidden.

This is not quite a complete Python implementation either. You'll need a copy of ArcGIS Pro if you want to replicate the first stage where the clipped country nightlight rasters are created. Though, I have included a few clipped country nightlight rasters for you to mess around with the sampling and understand the format. All you need for the rest is a copy of Python 3.7.

## Before you begin
You'll need a Google Static Maps API key to download the daytime satellite sensor imagery. One of these is easy to obtain through Google's cloud platform and the first 100,000 images you download each month are free (at the time of writing). That won't allow you to download 1000 images per country for free in a single month, but it will allow you to either get ~500 images per country or to get the images for half the countries. When you have a key, you should copy it into the text file `google_api_key.txt`. This file should be a single line containing only the key and nothing else.

You'll also need to download the World Bank National Accounts data files to obtain the GDP figures: `https://data.worldbank.org/indicator/NY.GDP.MKTP.CD`. In the paper we used the 2019 GDP figures in current US$, but you will probably want to use the most recent figures as the Google Static Maps API images are updated periodically (see section III.D.3 in the paper).

You will also need to download the trained CNN model from  J Mather's reimplementation of the Jean *et al.* 2016 paper which can be found here: `https://github.com/jmather625/predicting-poverty-replication#use-my-model`.

## Clipped country nightlight rasters
The VIIRS nighttime rasters can be found here: `https://payneinstitute.mines.edu/eog-2/viirs/`. Our sampling code uses a clipped version of these rasters for each country. If you have access to ArcGIS Pro then you can use the script from the file ArcGIS_clipping_script.py in the ArcGIS Pro python prompt to produce these clipped country nightlight rasters. The format of these rasters is that, within each country's borders, nightlight values are identical to the main VIIRS rasters but, outside each country's borders, each pixel has a value of -1. When imported, the -1 values are used as a mask to stop sampling outside a country's borders. The clipped country nightlight raster format to be used is `.tif` and the `.tif.xml` metadata files produced by ArcGIS Pro need to be in the same folder. These clipped rasters should use the espg 4326 coordinate system.

## Requirements
The Python version used for this implementation was Python 3.7. First thing you'll need to install is gdal; details for installing gdal for python can be found here: `https://gdal.org/api/python.html`. A list of the minimum packages you'll need to install can be found in `requirements.txt`. If you wish to reproduce Fig. 2 (the map of the UK sample) from the paper then you will also need to install the Python packages: shapely, Descartes, Geopandas.

## Run order
1. `ArcGIS_clipping_script.py` (if ArcGIS installed)
2. `Sample.py`
3. `Download_Sample_Images.py`
4. `Feature_Extraction.py`
5. `PCA.py`
6. `Model_Fitting.py`

## Model fitting
The search for the optimal number of regions isn't automatic. You'll have to run the Monte Carlo search with various parameters and so forth. This means that the exact results from the paper (i.e. the AICc scores) aren't perfectly reproducible but the optimal number of regions should remain consistent. I'll try and fix this in a future implementation.

 - Nathan Price 31.10.21
(email: n.price1@lancaster.ac.uk)