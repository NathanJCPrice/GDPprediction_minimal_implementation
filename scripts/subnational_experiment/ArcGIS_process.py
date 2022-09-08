# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:14:46 2022

@author: pricen1
"""

# -*- coding: utf-8 -*-
"""
A script for the ArcGIS Pro python interpreter to produce clipped country nightlight rasters. Some filepaths are hardcoded in to make users suffer. >:)
"""

import unicodedata, arcpy
import pandas as pd
import numpy as np

def remove_accents(input_str):
    #From: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def strip_specials(string):
    #I think I copied this from somewhere but I don't remember where.
    #If you know who this is attributable to, please email me.
    out = ""
    for x in string:
        if x.isalnum():
            out = out + x
    return out


root_dir = r'C:/Data/Nightlights/2020/CountryClips/'
VIIRS = "VNL_v2_npp_2020_global_vcmslcfg_c202101211500_average_tif" #The VIIRS raster layer name.

proj = r"C:\Data\GIS\USCountyGDP" #Project filepath.
arcpy.env.workspace = proj
layer = "USA_Counties" #The layer which contains the world border shapefiles. We used the Esri's World Countries (Generalized) dataset.
counties_save_path = root_dir+"countries.csv" #Location to output a csv which lists all the countries and their ISO and OID codes.

cursor = arcpy.da.SearchCursor(layer, ("NAME","STATE_NAME","FIPS","SHAPE@AREA","OID@"))#, sql_clause=(None, 'GROUP BY COUNTRYAFF'))

counties = [row for row in cursor]
counties = pd.DataFrame(counties,columns=['county','state','fips','area','OID'])
cursor.reset()


counties = counties.sample(n=200, random_state=56)

counties.to_csv(counties_save_path, sep=',')

for _, county in counties.iterrows():
    TEMP_LAYER = layer + "_" + county.county.replace(" ","") + "_TEMP2"
    sql_query = "NAME = \'" + county.county + "\' AND STATE_NAME = \'" + county.state +"\'"
    arcpy.Select_analysis(layer, TEMP_LAYER, sql_query)
    s = remove_accents(strip_specials(county.state)).decode('utf-8')
    c = remove_accents(strip_specials(county.county)).decode('utf-8')
    output = root_dir+ "Joined_VIIRS_" + s + "_" + c + ".tif"
    arcpy.Clip_management(in_raster=VIIRS, rectangle="", out_raster=output, in_template_dataset=TEMP_LAYER, nodata_value="-1", clipping_geometry="ClippingGeometry")
    arcpy.Delete_management(TEMP_LAYER)
    print(_)