# -*- coding: utf-8 -*-
"""
I wrote this class to handle the logic of looping through country data files.

Eventually I'll use it in every file, but for now it's only used in a few places.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import unicodedata

class CountryIterator:
    def __init__(self, countries, folder, file_pattern, library="numpy", output_filepath=False, individual_country_folders=False):
        self.countries = countries
        self._i = -1
        self._folder = folder
        self._file_pattern = file_pattern
        self.length = len(countries)
        self._library = library
        self._progress_bar = tqdm(total=self.length, position=0, leave=True)
        self._output_filepath = output_filepath
        self._individual_country_folders = individual_country_folders
        
    def __next__(self): #I need to come back and comment this properly.
        self._i += 1
        if self._i < self.length:
            country = self.countries[self._i]
            if not self._individual_country_folders:
                country = self.countries[self._i]
                file_path = self._folder + self._file_pattern.format(country)
            else:
                file_path = os.path.join(os.path.join(self._folder, country), self._file_pattern.format(country))
            if self._library=="numpy":
                data = np.genfromtxt(file_path, delimiter=',')
            elif self._library=="pandas":
                data = pd.read_csv(file_path, sep=',')
            else:
                raise Exception("Library not supported or unrecognised.")
            self._progress_bar.update()
            if not self._output_filepath:
                return (country, data)
            else:
                return (country, data, file_path)
        raise StopIteration
        
    def __iter__(self):
        return self
    
    @staticmethod
    def __remove_accents(input_str):
    #From: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii

    @staticmethod
    def __strip_specials(string): #Can't remember who this is attributable to.
        out = ""
        for x in string:
            if x.isalnum():
                out = out + x
        return out
    
    @staticmethod
    def __clean_country_names(df):
        for i, r in df.iterrows():
            df.at[i, 'country'] = CountryIterator.__remove_accents(CountryIterator.__strip_specials(df.at[i,'country'])).decode('utf-8')
    
    @staticmethod
    def get_country_codes(file_path, exclusions=[]):
        country_codes = pd.read_csv(file_path)
        country_codes2 = country_codes.copy()
        seen = exclusions
        for i, r in country_codes2.iterrows():
            if r.country in seen:
                country_codes.drop(i,axis=0,inplace=True)
            else:
                seen.append(r.country)
        print(country_codes)
        CountryIterator.__clean_country_names(country_codes)
        return country_codes.reset_index(drop=True)