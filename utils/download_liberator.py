import pandas as pd
import wget
import os
from tqdm import tqdm 

csv_data_fname = 'liberator_full_dataset.csv'# name of csv to read from (in same parent directory)
save_directory = 'full_dataset'  # directory name where data will be downloaded

# Read in the data, create directory to store dataset
df = pd.read_csv(csv_data_fname, nrows=5)
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
numrows = len(df.index)

# Iterate over rows, download all issues to their own directory
for index, row in tqdm(df.iterrows(), total=numrows, colour='green'):
    filename = row['image_id'] + '.jpeg'
    issue_directory = os.path.join(save_directory, row['issue_id'])
    if not os.path.exists(issue_directory):
        os.mkdir(issue_directory)
    filepath = os.path.join(issue_directory, filename)
    if not os.path.exists(filepath): #only download if its not there already
        wget.download(url=row['image_url'], out=filepath, bar=None)