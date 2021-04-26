# Standard data science import
import pandas as pd
# For making requests to LoC API
import requests
# Streamlit
import streamlit as st
# Spacy
import spacy
# Spacy visualization
from spacy import displacy
# Spacy + Streamlit integration
from spacy_streamlit import visualize_ner
# Pillow
from PIL import Image

import natas

from RawOCR_PostProcess import correct_raw_ocr 

nlp = spacy.load('en_core_web_sm')

def correct_raw_ocr(ocr_text):

    incorrect = []
    words = []
    
    doc = nlp(input_ocr_text)
    
    print("Iterating through words in text..")    
    for w in doc:
        word = w.text
        correct = natas.is_correctly_spelled(word)
        if not correct:
            print("Incorrect Spelled Word: " + word)
            incorrect.append(word)
        words.append(word)

    print("Generating replacement candidates...")
    candidates = natas.ocr_correct_words(incorrect)

    print("Using the top candidate for replacement in the input text...")

    for w,c in zip(incorrect, candidates):
        # Retrieve the top candidate 
        top = c[0]
        print("\tReplacing {" + w + "} " + "with {" + top + "}") 
        output = output.replace(w, top)


    print("\n\nORIGINAL\n")
    print(input_ocr_text)

    print("\n\nCORRECTED\n")
    print(output + "\n")
    
    return output

# Initialize messy variables here
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]

# Set title
st.title("Text Analysis Tool")

# Sample text to analyze (Liberator excerpt)
text = """
Following the Potsdm Declaration by Adolf Hitler and the German unconditional surrender on 8 May 1945.  the Allies on 26 July 1945 and the refusal of Japan to surrender on its terms, the United States dropped the first atomic bombs on the Japanese cities of Hiroshima, on 6 Aust 1945, and Nasaki, on 9 August. F on 9 August, Japan announced its intention to surrender on 15 August 1945, cementing total victory in Asia for the Allies. In the wake of the war, Germany and Japan were occupied, and war crimes tribuns were conducted gainst German and Japanese leaders. Despite their well documented war crimes, mainl perpetrated in Grecee and Yugoslavia, Italian leaders and generals were often pardoned, thanks to diplomatic activities."""

# Use small pre-trained NER base model from Spacy (move to transformers later)
# Visualize within Streamlit

text = correct_raw_ocr(text)

doc = nlp(text)
labels = nlp.get_pipe("ner").labels


visualize_ner(doc, labels=labels)

# Create the sidebar
st.sidebar.title("LoC Lookup")

# Get the metadata from the entities in the document
data = [
    [str(getattr(ent, attr)) for attr in NER_ATTRS]
    for ent in doc.ents
    if ent.label_ in labels
]

# Create a pandas DataFrame using above data
df = pd.DataFrame(data, columns=NER_ATTRS)

# Drop-down for recognized entity
entity_choice = st.sidebar.selectbox(
    'Entity Selection',
     df['text']
)

# Drop-down for whether you want to search Subjects or Names on LoC
loc_search_choice = st.sidebar.selectbox(
    'LoC Search Choice',
     ['Subject', 'Name']
)

# Craft a API search query based on selections and return the JSON
url = 'https://www.loc.gov/search/?q={}&fo=json'.format(entity_choice)

# Get just the titles of each link in the JSON
titles = [t['title'] for t in requests.get(url).json()['results']]

# Populate the choices with LoC search results
loc_choice = st.sidebar.selectbox('Results', titles)
