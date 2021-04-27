# CS501-BPL-Liberator
Repository containing code related to Boston University's Spark ML Practicum (CS 501 T2) collaboration with Boston Public Library.

<img src="https://www.bu.edu/spark/files/2017/04/spark-logo-round.png" height="100"/> <img src="https://cor-liv-cdn-static.bibliocommons.com/images/MA-BOSTON-BRANCH/logo.png?1613638160260" width="200"/> 

# Introduction

## Project Overview and Goal

  *The Liberator* is a 19th-century abolitionist and womens' rights newspaper, founded and published by the abolitionist William Lloyd Garrison. The Boston Public Library's Digital Repository houses a full scanned archive of the newspaper, and this project's goal is to digitize and digest the articles and their text. Existing optical character recognition (OCR) models would interpret a newspaper page as one piece of text and attempt to read it from the left edge to the right edge before wrapping around again. This would produce nonsensical output for our input data. To leverage existing OCR solutions, we implement a multi-stage pipeline, consisting of article segmentation, OCR of text, and named entity recognition (NER) of the extracted article-level text.
  
  ## Why Machine Learning?
  
  This project and its requirements naturally lend themselves to the advantages of a machine learning approach. OCR and NER tools have been widely improved by machine learning approaches, and there exist vast libraries and toolkits to aid in the simple out-of-the-box development of custom OCR/NER pipelines, some of which will be discussed here. Further, the problem of intelligent and accurate article segmentation necessitates the use of machine learning, specifically that of neural networks. In our problem, as is commonly the case with historic newspaper digitization work, the Liberator dataset is not a perfectly-scanned digitization. Many misaligned scans, damaged papers, and blemished pages throw off any direct segmentation approach. Furthermore, over the 30+ year run of The Liberator, the paper underwent constant design and layout changes, making intelligent article extraction a necessity.

## Birds-Eye-View of Pipeline

<img src="./media/diagram.png" width="800"/>

# Getting Started

## How it Works

The entire pipeline looks for input images in either a directory provided as an argument, or by looking inside of data/input_images if no argument is provided. By simply running main.py without arguments, the entire pipeline will be ran on the provided test files inside of input_images. This is a convenient way to familiarize oneself with the function of the app and also for grading. Otherwise, a full absolute path to a image directory that follows the provided directory organization scheme is also doable.

## Dependencies
All Python dependendicies for this pipeline are handled with the Pipfile. Usage of the Pipfile and pipenv is explained in more detail below.

More pertinently, there are a few packages and files that will need to be manually downloaded and placed in order to run the full pipeline.

1. bbz-segment/05_predicition/**data/models** - Inside of the subdirectory bbz-segment/05_prediction, it is required to create the subdirectory **data/** and place the models folder containing the pre-trained ML models for article segmentation. This pre-trained TensorFlow model is available [here on Dropbox](https://www.dropbox.com/sh/7tph1tzscw3cb8r/AAA9WxhqoKJu9jLfVU5GqgkFa?dl=0), by the original authors.

2. ner/**stanza_resources** - The NER portion of the pipeline uses Stanza, an NLP package by the Stanford NLP Group. stanza-resources/ contains the language processors required to process and tag entities in text. The folder can be found [here on Google Drive](https://drive.google.com/drive/folders/1Le0sxSRzmzdCAIeZRRKjs9mjaD-VCmcd?usp=sharing) and should be placed inside of the directory ner/.

3. **config/credentials.json** - For the OCR, we utilize Google Cloud Vision. Accessing the Google CV API requires setting up a service account and setting the proper environment variable to point to your credentials.json file, containing the API Key information. Our pipeline automatically checks and sets the appropriate environment variable to point to a credentials.json file inside of a **config** directory. All that needs to be done is to create a config/ directory in the main project directory and place the Google CV credentials.json (with that same filename) inside of it. Detailed steps to create a service account and download a credentials.json with your API Key can be found here: https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account.

Without a credentials.json, the pipeline will still run up to article segmentation, and then announce that it is quitting the rest of the pipeline.

## Run the Pipeline!

This section shares step-by-step instructions on running the pipeline.

Get the repo with `git clone https://github.com/SikandAlex/CS501-BPL-Liberator.git`

Next, using the information under the **Dependencies** section, ensure each of the necessary packages are properly placed.

Then, install Python dependencies. Use `pip3 install pipenv` to get pipenv first, then, from __inside of the project root directory__:
1. `pipenv install`
2. `pipenv shell`

Now you should have all of the dependencies installed.

Finally, you are ready to run. To run with the provided test files inside of data/input_images/, just run with `python main.py`. To run on another directory,
run with `python main.py -i <absolute path to input directory>`. The input directory you are providing must follow the same issue organization and naming schema
as data/input_images/. Please read the below Notes and Considerations before running!

## Notes and Considerations

This pipeline is heavy in computing power and time. Running the three-issue test set through the entire pipeline on a 2015 Macbook Pro, Intel i5 2.7 GHz processor took about ~15 minutes and a lot of whirring. For grading or testing purposes, it would be okay to remove two full issues from input_images/. This is not necessary, however.

## Data Downloader/Input Format

Inside of utils, there is a data_downloader.py and a data_organization.py file. These are the scripts used on BU's Shared Computing Cluster to download and organize The Liberator dataset for our use. While they are tailored to run on the SCC, one can use and edit these if one wanted to download or save part of the dataset (the full dataset is considerably large!).

# Detailed Overview

## Column Extractor

## Article Segmentation

## OCR & NER
The articles crops are passed to Google Cloud Vision Document Text Detection API and we retrieve the outputted text. If the OCR model produces mispelled words or isn't as 'clean' as we would like, we pass the raw OCR output to one of two spell-correction libraries natas / autocorrect as form-data to a Flask API. This increases the quality of the OCR results. 

After the text has been extracted and cleaned up, we fine-tuned a spacy en_core_web_lg model with streamlit to detect several entities of interest to the Boston Public Library. We intend to implement additional rule-based and dictionary-based approaches to increase the accuracy of the NER model.  

# Further Work

# References and Attributions
