import pandas as pd
import wget
import os
from tqdm import tqdm 

"""
This python file downloads The Liberator dataset based off 
issue id, image id and file URLs from a CSV file. 

NOTES:
By default when you run this file it will download the full dataset (~7,500 images/40 GB) to 
a directory on the same level called 'full_dataset'. 
Within this directory will be a directory for each issue, named after its issue ID. 
Images will be saved to the correct issue directory, about 4 images per issue. 
if images are already present in the correct issue directory, they will not be downloaded twice.

USAGE:
- If you dont want to download the full dataset, change num_pages to the number of images you want to download
    - if you want to download the full dataset later these images wont be downloaded twice 
- Change csv_data_fname to the csv to read from if neccessary
- Change save_directory to the name of the directory to download to (not reccomended)
"""

csv_data_fname = 'data/liberator_full_dataset.csv'# name of csv to read from (in same parent directory)
save_directory = 'data/full_dataset'  # directory name where data will be downloaded
num_pages = 5 # change this to a non zero value to download only a certain number of pages. (useful for testing)

# Read in the data, create directory to store dataset
if num_pages:
    df = pd.read_csv(csv_data_fname, nrows=num_pages)
else:
    df = pd.read_csv(csv_data_fname)
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
numrows = len(df.index)

# Iterate over rows, download all issues to their own directory
for index, row in tqdm(df.iterrows(), total=numrows, colour='green'):
    filename = row["filename"]
    issue_directory = os.path.join(save_directory, row['issue_id'])
    if not os.path.exists(issue_directory):
        os.mkdir(issue_directory)
    filepath = os.path.join(issue_directory, filename)
    if not os.path.exists(filepath): #only download if its not there already
        wget.download(url=row['image_url'], out=filepath, bar=None)