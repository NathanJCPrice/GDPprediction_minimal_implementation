# -*- coding: utf-8 -*-
"""
This class handles all the importance sampling logic. See paper for detailed description.

This whole class assumes epsg 4326.
"""

from PIL import Image
from PIL.TiffTags import TAGS
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from itertools import product

class GeoSampler:
    
    def __init__(self, fpath):
        self.__fpath__ = fpath
        self.raster = geoio.GeoImage(fpath)
        self.pixels = self.raster.get_data()[0] #This variable stores the pixels as a numpy array.
        self.selection_probabilities = None #These need to be initialised by the initialise_selection_probabilities method.
        self.weights = None #These weights are the weights used in the Monte Carlo estimator.
        self.inbounds = (self.pixels>=0) #Boolean array saying whether each pixel is in the country or not.
        self.sample_pixels = None #The pixels that are chosen to be sampled.
        self.sample_points = None #The lat-lon sample points.
        self.shape = self.pixels.shape
        
        XML = ET.parse(fpath + ".xml") #The .tif.xml file contains the bounding box of the geotiff which is needed for sampling.
        root = XML.getroot()
        bndsXML = [i for i in root.iter('GeoBndBox')][0]
        bndsXML = {i.tag:i.text for i in bndsXML.iter()}
        self.bounds = {key:float(bndsXML.get(key)) for key in ['westBL', 'eastBL', 'northBL', 'southBL']}
        

    def __repr__(self):
        self.raster.show()
        return "Working with raster: {}".format(self.__fpath__)
    
    def raster_to_proj(self, r,c): #Utility function which converts an image pixel location to the associated lat-lon coordinates.
        projx = self.bounds['westBL']+(self.bounds['eastBL']-self.bounds['westBL'])*c/self.pixels.shape[1]
        projy = self.bounds['northBL']+(self.bounds['southBL']-self.bounds['northBL'])*r/self.pixels.shape[0]
        return (projx, projy)
    
    def proj_to_raster(self, projx, projy): #And vice versa.
        c = (projx-self.bounds['westBL'])*self.pixels.shape[1]/(self.bounds['eastBL']-self.bounds['westBL'])
        r = (projy-self.bounds['northBL'])*self.pixels.shape[0]/(self.bounds['southBL']-self.bounds['northBL'])
        return (r, c)
        
    
    def initialise_selection_probabilities(self, minimal_spoof_luminosity, correct_for_curvature=False): #minimal_spoof_luminosity is the same as epsilon. See paper.
        self.pixels[self.inbounds] = np.maximum(minimal_spoof_luminosity, self.pixels[self.inbounds])
        p = np.copy(self.pixels) #That memory usage is painful. I need to rewrite this.
        if correct_for_curvature: #The earth is a globe but flatearthers can turn off the curvature correction if they so please.
            for i in range(0, p.shape[0]):
                lat = np.pi*self.raster_to_proj(i, 0)[1]/180
                p[i] *= np.cos(lat) #Assumes the earth is a sphere. The discrepancy is negligible for most every country.
        K = sum(p[self.inbounds]) 
        p[self.inbounds] /= K
        p[~self.inbounds] = -1
        self.selection_probabilities = p
        return None
    
    def PPS_sample(self, n, progress_bar=False): #This is the first stage of the sampling scheme. See paper.
        if self.selection_probabilities is None:
            raise Exception("Selection probabilities uninitialised.")
        
        times_sampled = np.zeros(self.pixels.shape, dtype='uint')
        indexes = np.where(self.inbounds)
        a = np.array(self.selection_probabilities[self.inbounds])
        a = np.append(0,a)
        c = np.cumsum(a) #I wonder how many people have giggled at this function name through history.
        x = np.linspace(0,n-1,n)/n
        x += np.random.random(1)/n
        t = 0
        if not progress_bar:
            for j in x: #Pretty standard systematic PPS sampling implementation. Surely a faster method must exist?
                m = np.argmax(c >= j)-1
                t = t + m
                c = c[m:]
                t0 = (indexes[0][t],indexes[1][t])
                times_sampled[t0] += 1
        else:
            for j in tqdm(x, total=len(x), position=0, leave=True):
                m = np.argmax(c >= j)-1
                t = t + m
                c = c[m:]
                t0 = (indexes[0][t],indexes[1][t])
                times_sampled[t0] += 1
        self.weights = 1/(n*self.selection_probabilities)
        self.sample_pixels = times_sampled
        return None
    
    def set_weights(self, n):
        if self.selection_probabilities is None:
            raise Exception("Selection probabilities uninitialised.")
        self.weights = 1/(n*self.selection_probabilities)
        return None
    
    def get_weight_at_proj_pos(self, df):
        if self.selection_probabilities is None:
            raise Exception("Selection probabilities uninitialised.")
        r, c = self.proj_to_raster(df['lons'],df['lats'])
        r = r.to_numpy('int')
        c = c.to_numpy('int')
        return self.weights[r,c]
    
    def get_value_at_proj_pos(self, df):
        if self.selection_probabilities is None:
            raise Exception("Selection probabilities uninitialised.")
        r, c = self.proj_to_raster(df['lons'],df['lats'])
        r = r.to_numpy('int')
        c = c.to_numpy('int')
        return self.pixels[r,c]
    
    def get_sample_points(self):
        if self.sample_pixels is None:
            raise Exception("No sample yet drawn.")
        point_sample = np.empty((2,np.sum(self.sample_pixels)))
        row_counter = 0
        xiter = product(range(0,self.shape[0]),range(0,self.shape[1]))
        l = self.shape[0]*self.shape[1]
        for r, c in tqdm(xiter, total=l, position=0, leave=True):
            p = self.sample_pixels[r,c] #Important to realise that this can be >0. Multiple points are often selected in urban pixels.
            if (p==0):
                continue
            noise = np.random.random((2,p)) #The probability should not actually be uniform. Correct if time permits.
            pt0 = self.raster_to_proj(r,c)
            pt1 = self.raster_to_proj(r+1,c+1)
            point_sample[1,row_counter:row_counter+p] = pt0[0]*noise[0] + pt1[0]*(1-noise[0])
            point_sample[0,row_counter:row_counter+p] = pt0[1]*(1-noise[1]) + pt1[1]*noise[1]
            row_counter += p
        self.sample_points = point_sample
            
    def clean_up_and_close(): #Complete when time permits.
        return None
    
    def visualise_geobounds(self):
        return Image.fromarray(np.round(self.inbounds*255).astype(np.uint8),mode='L')
    
    def visualise_sample(self):
        if self.sample_pixels is None:
            raise Exception("No sample yet drawn.")
        
        scale = 255/np.max(self.sample_pixels)
        print(scale)
        
        return Image.fromarray(np.round(self.sample_pixels*scale).astype(np.uint8),mode='L')
