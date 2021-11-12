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


root_dir = 'D:\\GDP_prediction_minimal_implementation\\'
VIIRS = "Joined_VIIRS" #The VIIRS raster layer name.

proj = "D:\\ArcGIS\\VIIRS_project.gdb" #Project filepath.
arcpy.env.workspace = proj
layer = "country_borders" #The layer which contains the world border shapefiles. We used the Esri's World Countries (Generalized) dataset.
countries_save_path = root_dir+"countries.csv" #Location to output a csv which lists all the countries and their ISO and OID codes.

cursor = arcpy.da.SearchCursor(layer, ("COUNTRYAFF","ISO_3DIGIT","SHAPE@AREA","OID@"))#, sql_clause=(None, 'GROUP BY COUNTRYAFF'))

countries = [row for row in cursor]
countries = pd.DataFrame(countries,columns=['country','country_code','area','OID'])
cursor.reset()

countries.to_csv(countries_save_path, sep=',')

for country in countries.country.unique():
    temp = countries.loc[countries.country==country]
    max_area = np.max(temp.area)
    country_string = remove_accents(strip_specials(country)).decode('utf-8')
    country_code = temp.iloc[0]['country_code']
    TEMP_LAYER = layer + "_" + country_code + "_TEMP2"
    #The following query removes small overseas territories. 
    #These contribute minimally to GDP so it has no effect to the final model's predictive power.
    #If you leave the overseas territories in you get HUGE rasters that are a pain.
    if not country_string == "CotedIvoire": #The punctuation in this country name is a nuisance.
        sql_query = "COUNTRYAFF = \'" + country + "\' AND Shape__Area > " + str(int(max_area/20))
    else:
        sql_query = "ISO_3DIGIT = \'" + "CIV" + "\' AND Shape__Area > " + str(int(max_area/20))
    arcpy.Select_analysis(layer, TEMP_LAYER, sql_query)
    output = root_dir+"country_nightlight_rasters\\" + VIIRS + "_" + country_string + ".tif" #Ew.
    arcpy.Clip_management(in_raster=VIIRS, rectangle="", out_raster=output, in_template_dataset=TEMP_LAYER, nodata_value="-1", clipping_geometry="ClippingGeometry")
    arcpy.Delete_management(TEMP_LAYER)
    print(country)

#Creates a country .txt list.
#The reason there's a seperate .txt list of countries is because of legacy nonsense.
with open(root_dir+"country_list.txt","x") as country_list: #Ew.
    for country in countries.country.unique():
        to_write = remove_accents(strip_specials(country)).decode('utf-8') + "\n"
        country_list.write(to_write)
    country_list.close()