'''
Very simple download interface to download images from Google's Static Maps API
A modified version of: https://github.com/jmather625/predicting-poverty-replication/blob/master/utils/google_downloader.py
'''
import requests
import matplotlib.pyplot as plt
from io import BytesIO

class GoogleDownloader:
    def __init__(self, access_token):
        self.access_token = access_token
        self.url = 'https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size={}x{}&maptype=satellite&key={}'
    
    def download(self, lat, long, zoom=14, width=400, height=400, crop=False):
        if (crop): #Gotta get rid of that Google logo.
            height = height+50
        res = requests.get(self.url.format(lat, long, zoom, width, height, self.access_token))
        # server needs to make image available, takes a few seconds
        if res.status_code == 403:
            return 'RETRY'
        assert res.status_code < 400, print(f'Error - failed to download {lat}, {long}, {zoom}')
        image = plt.imread(BytesIO(res.content))
        if (crop):
            image = image[25:height-25] #Cropping the Google logo out.
        return image
    