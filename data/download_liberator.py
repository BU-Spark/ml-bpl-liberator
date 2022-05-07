import pandas as pd
import wget
import os
from tqdm import tqdm 

csv_data_fname = 'liberator_full_dataset.csv'# name of csv to read from (in same parent directory)
save_directory = 'full_dataset'  # directory name where data will be downloaded

# Read in the data, create directory to store dataset
df = pd.read_csv(csv_data_fname, nrows=50)
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
numrows = len(df.index)

# progess bar for pandas
tqdm.pandas()

# Iterate over rows, download all issues to their own directory
def get_and_save_image(row):
    filename= row["filename"]
    issue_directory = os.path.join(save_directory, row['issue_id'])
    if not os.path.exists(issue_directory):
        os.mkdir(issue_directory)
    filepath = os.path.join(issue_directory, filename)
    if not os.path.exists(filepath): #only download if its not there already
        wget.download(url=row['image_url'], out=filepath, bar=None)

df.progress_apply(get_and_save_image, axis=1)