import cv2
import numpy as np 
import json
import sys
import os
from pathlib import Path
from colorama import Fore, init
init(autoreset=True)
from google.cloud import vision_v1 as vision
import io
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import definitions
from ner.stanza_ner import StanzaNER

# This environment variable needs to be set with your credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(definitions.CONFIG_DIR, "credentials.json")

def crop_article(filename, x1, y1, x2, y2, article_counter, image_counter, issue, path=definitions.CROPPED_ARTICLES):
    '''
    Function to produce single cropping of an article given coordinates
    '''

    img_path = os.path.join(definitions.INPUT_DIR, issue, filename)
    im = cv2.imread(img_path)
    
    os.makedirs(os.path.join(path, issue, str(article_counter)), exist_ok=True)
    articlePath = Path(os.path.join(path, issue, str(article_counter))).joinpath("article_"+str(article_counter)+"_" + str(image_counter) +".png")
    
    roi = im[x1:x2,y1:y2]
    if not cv2.imwrite(str(articlePath), roi):
        print("crop failure on " + filename)



def crop_articles(jsonData):
    '''
    Using the loaded JSON data from file produced by the article segmentation code, this method crops the articles 
    and saves their pathfiles into a JSON that the OCR method can then read

    Returns the issue's croppings subdirectory path
    '''

    article_counter = 1
    issue = jsonData[0]['issue_id']

    for article in jsonData:
        image_counter = 1
        for image in article['images']:

            #skip and throw away small "articles", i.e. false positives
            if image['coordinates'][2] - image['coordinates'][0] < 200:
                continue

            crop_article(image['filename'], image['coordinates'][0], image['coordinates'][1], image['coordinates'][2], image['coordinates'][3], article_counter, image_counter, issue)
            image_counter += 1
        article_counter += 1

    return os.path.join(definitions.CROPPED_ARTICLES, issue)

def articleOCR(file_name):
    '''
    Given a cropped article, runs through Google Cloud Vision OCR
    '''
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
        
    image = vision.Image(content=content)

    # Hit the API and get back the text annotations (description is the good part)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception('{}'.format(response.error.message))
        
    texts = response.text_annotations

    if(texts != []):
        return texts[0].description.strip(), True
    else:
        return "", False

'''
The following three methods are helper functions to organize and write the final JSON Output file
'''

# Takes info for each article and creates a JSON dictionary
def article_to_dict(section="null",title="null",images="null", ocr_text="null", named_entities="null"):
    data = {
        "section": section,
        "title": title,
        "images": images,
        "ocr_text": ocr_text,
        "named_entities": named_entities
    }
    
    return data

# Maps article JSON dictionaries to their issue id
def articles_to_issue(issue_id, articles):
    data = {
        "issue_id": issue_id,
        "articles": articles
    }
    return data

# Appends the given issue and its article information to data.json
def add_issue_to_json(issueID, issue_data):
    with open(Path(definitions.JSON_OUTPUT).joinpath('data.json'), 'r+') as file:
        print(Fore.CYAN + "Writing final output to data.json for issue " + issueID)
        data = json.load(file)
        temp = data['issues']
        temp.append(issue_data)
        file.seek(0)
        json.dump(data, file, indent=4)


def get_title_from_ocr_text(text):
    '''
    We did not fully build out title-extraction functionality.
    Many articles start with a title in ALL CAPS or have a period at the end of the title. 
    We can utilize this information to gather the article titles given the ocr text.
    '''

    firstLine = text.split("\n")[0]
    if firstLine.isupper() or (firstLine.endswith(".") and len(firstLine) <= 25):
        return firstLine

    findPeriod = firstLine.find(".")
    if findPeriod > 0 and findPeriod <= 25:
        return firstLine[:findPeriod]

    return "null"


def issue_ocr(segment_json_path, NER_PIPELINE):
    '''
    This method does a lot - given a path to a JSON that encodes article segmentation information,
    this method will (1) crop and write all of the articles, (2) run all articles through OCR,
    (3) run the text through the Stanza NER inside of /ner (4) delete all croppings, and finally,
    (5) write the issue information to the final data.json product.
    Args:
    segment_json_path: path to the article segmentation JSON (in data/segement_outputs)
    NER_PIPELINE: the StanzaNER Object in ner/StanzaNER.py. This is instantiated and passed
    from main.py so that the models are only loaded once in the entire pipeline.
    '''

    # Open and load article segmentation info
    f = open(segment_json_path)
    json_data = json.load(f)
    cur_issue_id = json_data[0]['issue_id']

    # Crop articles and save to data/cropped_articles/[issue_id]
    print("Cropping and extracting text & entities from " + cur_issue_id)
    crop_path = crop_articles(json_data)

    #List of article JSON objects
    articles = []

    # Walk the cropped_articles directory for this issue
    for root, subdirectories, _ in os.walk(crop_path):
        for subd in sorted(subdirectories): 
            full_article_text = [] # list to maintain article text if multiple images
            for article in sorted(os.listdir(os.path.join(root, subd))): 
                if article.endswith(".png"): # assertion
                    article_path = os.path.join(root, subd, article)
                    article_index = int(subd) - 1 # index of the article in the segment JSON file, used to grab image information

                    article_text, found = articleOCR(article_path) # Runs the OCR on articles!
                    if found: # If no text is found, this image is skipped
                        full_article_text.append(article_text)

                # print("Deleting file " + article)
                os.remove(article_path)

            if full_article_text:
                ocr_text = " ".join(full_article_text) # Concatenates text for multi-image articles
                title = get_title_from_ocr_text(ocr_text)

                ner_info = NER_PIPELINE.run_ner(ocr_text) # Runs NER on extracted text!

                a = article_to_dict(title=title, images=json_data[article_index]["images"], ocr_text=ocr_text, named_entities=ner_info)
                articles.append(a)

            # print("Deleting subdirectory " + root + "/" + subd)
            os.rmdir(os.path.join(root, subd))

        print("Deleting temp cropping subdirectory " + root)
        os.rmdir(root)
    
    issueData = articles_to_issue(cur_issue_id, articles)
    add_issue_to_json(cur_issue_id, issueData)
