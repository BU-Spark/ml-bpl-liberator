import pandas as pd
import wget
import os
import shutil
# Read in the data I've assembled
df = pd.read_csv('liberator-data.csv')
# Put everything in a liberator-data subfolder
base = './liberator-data/'
# Create liberator-data subfolder if necessary
if not os.path.exists(base):
    os.mkdir(base)
# Create necessary subdirectories named after id of the media
for id_,url in zip(df.id,df.dl_url):
    # Directory to create / place files in
    new_dir = base+id_
    # Create directory if necessary
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    # Name out output zip file to download (matches parent folder id)
    zip_loc = new_dir+'/'+id_+'.zip'
    # Download the zip file and extract it if we don't have it already
    if not os.path.exists(zip_loc):
        wget.download(url, zip_loc)
        shutil.unpack_archive(zip_loc, new_dir+'/data', 'zip')
