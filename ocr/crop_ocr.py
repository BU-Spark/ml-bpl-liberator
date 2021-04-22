import cv2
import numpy as np 
import json
import sys
import os
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import definitions


def crop_article(filename, x1, y1, x2, y2, article_counter, image_counter, issue, path=definitions.CROPPED_ARTICLES):
    img_path = os.path.join(definitions.INPUT_DIR, issue, filename)
    im = cv2.imread(img_path)
    
    os.makedirs(os.path.join(path, issue), exist_ok=True)
    articlePath = Path(os.path.join(path, issue)).joinpath("article_"+str(article_counter)+"_" + str(image_counter) +".png")
    
    roi = im[x1:x2,y1:y2]
    if not cv2.imwrite(str(articlePath), roi):
        print("crop failure on " + filename)


#Using the JSON file produced by the article segmentation code, this method crops the articles and saves their pathfiles into a JSON that the OCR method can then read
def crop_articles(segmentJSONPath):
    f = open(segmentJSONPath)
    jsonData = json.load(f)
    article_counter = 1
    issue = jsonData[0]['issue_id']

    for article in jsonData:
        image_counter = 1
        for image in article['images']:

            #skip small "articles", i.e. false positives
            if image['coordinates'][2] - image['coordinates'][0] < 200:
                continue

            crop_article(image['filename'], image['coordinates'][0], image['coordinates'][1], image['coordinates'][2], image['coordinates'][3], article_counter, image_counter, issue)
            image_counter += 1
        article_counter += 1



    
