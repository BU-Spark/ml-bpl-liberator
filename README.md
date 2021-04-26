# CS501-BPL-Liberator
Repository containing code related to Boston University's Spark ML Practicum (CS 501 T2) collaboration with Boston Public Library.

<img src="https://www.bu.edu/spark/files/2017/04/spark-logo-round.png" height="100"/> <img src="https://cor-liv-cdn-static.bibliocommons.com/images/MA-BOSTON-BRANCH/logo.png?1613638160260" width="200"/> 

# Introduction

## Project Overview and Goal

  *The Liberator* is a 19th-century abolitionist and womens' rights newspaper, founded and published by the abolitionist William Lloyd Garrison. The Boston Public Library's Digital Repository houses a full scanned archive of the newspaper, and this project's goal is to digitize and digest the articles and their text. Existing OCR models would interpret a newspaper page as one piece of text and attempt to read it from the left edge to the right edge before wrapping around again. This would produce nonsensical output for our input data. To leverage existing OCR solutions, we implement a multi-stage process, consisting of article segmentation, optical character recognition (OCR) of text, and named entity recognition (NER) of the extracted article-level text.
  
  ## Why Machine Learning?
  
  This project and its requirements naturally lend themselves to the advantages of a machine learning approach. OCR and NER tools have been widely improved by machine learning approaches, and there exist vast libraries and toolkits to aid in the simple out-of-the-box development of custom OCR/NER pipelines, some of which will be discussed here. Further, the problem of intelligent and accurate article segmentation necessitates the use of machine learning, specifically that of neural networks. In our problem, as is commonly the case with historic newspaper digitization work, the Liberator dataset is not a perfectly-scanned digitization. Many misaligned scans, damaged papers, and blemished pages throw off any direct segmentation approach. Furthermore, over the 30+ year run of The Liberator, the paper underwent constant design and layout changes, making intelligent article extraction a necessity.

## Birds-Eye-View of Pipeline

<img src="./media/diagram.png" width="800"/>

# Getting Started

## How it Works

## Dependencies
For Flask API run in a conda environment with python=3.6 with natas and spacy 

## Run the Pipeline!

## Notes and Considerations

## Data Downloader/Input Format

# Detailed Overview

## Column Extractor

## Article Segmentation

## OCR & NER
The articles crops are passed to Google Cloud Vision Document Text Detection API and we retrieve the outputted text. If the OCR model produces mispelled words or isn't as 'clean' as we would like, we pass the raw OCR output to one of two spell-correction libraries natas / autocorrect as form-data to a Flask API. This increases the quality of the OCR results. 

After the text has been extracted and cleaned up, we fine-tuned a spacy en_core_web_lg model with streamlit to detect several entities of interest to the Boston Public Library. We intend to implement additional rule-based and dictionary-based approaches to increase the accuracy of the NER model.  

# Further Work

# Downloading Data
Download data using the ```./utils/download-liberator.py``` and the included CSV file 

Currently this downloads ALL of the data, which may quickly eliminate all your free space (needs editing).

# Pipeline Overview 
Full pipleline demonstration in [PipelineExample.ipynb](https://github.com/SikandAlex/CS501-BPL-Liberator/blob/main/ocr/PipelineExample.ipynb)

## Column Detection (Using Frequency Information)
Pipeline capable of [detecting columns](https://github.com/jscancella/NYTribuneOCRExperiments/blob/master/findText_usingSums.py) from newspaper scans without using any deep learning (using frequency information) in OpenCV

<img src="./ocr/tester-contours.jpeg" width="400"/>

## OCR (Optical Character Recognition) 
Google Cloud Vision - [Document Text Detection](https://cloud.google.com/vision/docs/ocr)

<img src="./media/google-cloud-vision-ocr.png" height="250"/>

## Spell Correction
[autocorrect](https://github.com/fsondej/autocorrect) Python library

## OOTB NER
```python -m spacy download (en_core_web_sm or en_core_web_trf)```
### (NO GPU)
```spacy.load('en_core_web_sm')``` (small model)
### (GPU)
```spacy.load('en_core_web_trf')``` (transformer based model)

# Object Detection 
Proposed faster-rcnn model to extract article start/stop markers from the newspaper
[faster-rcnn implementation](https://colab.research.google.com/github/a8252525/detectron2_example_PCBdata/blob/master/PCBdata_fasterRCNN_colab.ipynb#scrollTo=WyR8yIqPFcNn)

## Extracting Articles from Column Image OCR
Detect titles + start/end article markers using a 1-3 class object detection model

## Labeling Data for Object Detection (not completed)
1) Label the data with bounding boxes using [Label Studio](labelstud.io)
2) Export in COCO format 

# Prototype NER (Named Entity Detection) LoC Entity Labeling Tool 

### Starting the web-app
streamlit run ```./ner/app.py```

Assuming that we need to know more than just the type of entity, we also need to assign the entity to a Library of Congress identifier. For this purpose, we have created a basic web application using [Streamlit](streamlit.io). 

<img src="./media/web-app.png" height="500"/>

The sample text is hard-coded for now but you can select one of the recognized entities in the sidebar and attempt to retrieve the top search results from the Loc API. 

To Do: It needs to be better and more tightly integrated with the LoC API. In addition, currently, when fetching data, the UI hangs because it's not being done asynchronously or something so that needs to be handled. We also need a submit button and a way to somehow save the labels to a data structure or file. 

<img src="./media/loc-retrieve.png" height="400"/>

