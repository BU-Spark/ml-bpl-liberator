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

# Initialize messy variables here
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]

img = Image.open("bpl-logo.png")
st.image(img)

# Set header
st.header("Text Analysis Tool")

# Sample text to analyze (Liberator excerpt)
text = "A large number of the colored citizens of Rochester having convened themselves together, for the important object of taking into consideration the anti-republican principles of the American Colonization Society, the Rev. Mr. Johnson was called to the chair, and Mr A. Lawrence was appointed Secretary. The meeting was then briefly addressed by the Secretary as follows: Countrymen and Brothers — When viewing the inhumanity and anti-christian principles of the American Colonization Society, in plotting our removal to Africa, (which is unknown to us as our native country,) it seems as though we were called upon publicly to express our feelings on the subject."

# Use small pre-trained NER base model from Spacy (move to transformers later)
# Visualize within Streamlit
nlp = spacy.load('en_core_web_sm')
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
