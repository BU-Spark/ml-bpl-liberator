# CS501-BPL-Liberator
Repository containing code related to Boston University's Spark ML Practicum (CS 501 T2) collaboration with Boston Public Library.

<img src="https://www.bu.edu/spark/files/2017/04/spark-logo-round.png" height="100"/> <img src="https://cor-liv-cdn-static.bibliocommons.com/images/MA-BOSTON-BRANCH/logo.png?1613638160260" width="200"/> 

## Downloading and Labeling Data
1) Download data using the ```./utils/download-liberator.py``` and the included CSV file (NEEDS EDITING, DOWNLOADS TOO MUCH DATA!!!)
2) Label the data with bounding boxes using [Label Studio](labelstud.io)
3) Export in COCO format 

## Object Detection 
Empty until I put together the faster-rcnn code adapted from [here](https://colab.research.google.com/github/a8252525/detectron2_example_PCBdata/blob/master/PCBdata_fasterRCNN_colab.ipynb#scrollTo=WyR8yIqPFcNn)

## OCR (Optical Character Recognition)
Take a look at the adaptive thresholding code for now 
![]('./media/thresh.png')

Ideas 


## NER (Named Entity Detection)
Assuming that we need to know more than just the type of entity, we also need to assign the entity to a Library of Congress identifier. For this purpose, we have created a basic web application using [Streamlit](streamlit.io). 

![]('./media/web-app.png')

The sample text is hard-coded for now but you can select one of the recognized entities in the sidebar and attempt to retrieve the top search results from the Loc API. 

![]('./media/loc-retrieve.png')

