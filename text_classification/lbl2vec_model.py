import json
import pandas as pd
from lbl2vec import Lbl2Vec
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import strip_tags

def tokenize(doc):
    '''
    doc: document text string
    returns: tokenized document
    strip_tags removes meta tags from the text
    simple preprocess converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long 
    simple preprocess also removes numerical values as well as punktuation characters
    '''
    return simple_preprocess(strip_tags(doc), deacc=True, min_len=2, max_len=15)

def get_keywords_dataframe(keywords_csv_path="text_classification/keywords.csv"):
    '''
    keywords_csv_path: the path to the csv representing the keywords for each subject
    returns: the dataframe with the necessary colunns to train the lbl2vec model
    '''
    
    labels = pd.read_csv(keywords_csv_path)

    # split keywords by separator and save them as array
    labels['keywords'] = labels['Synonyms'].apply(lambda x: x.split(';'))
    del labels['Synonyms']

    # convert description keywords to lowercase
    labels['keywords'] = labels['keywords'].apply(lambda description_keywords: [keyword.lower() for keyword in description_keywords])

    # get number of keywords for each class
    labels['number_of_keywords'] = labels['keywords'].apply(lambda row: len(row))

    return labels

def generate_title_text_df(input_json):
    '''
    input_data_path: the filepath to the json produced by the previous steps of the pipeline
    returns: a pandas dataframe with each row representing an article, with "title" "text" as columns
    '''

    issues = input_json["issues"]

    # Create the title/text dataframe
    # Iterate through the json, get the texts and titles
    texts = []
    titles = []

    for issue_number, issue in enumerate(issues):
        articles = issue["articles"]
        for article_number, article in enumerate(articles):
            text = article["ocr_text"]
            title = article["title"]
            texts.append(text)
            titles.append(title)

    # Generate the dataframe
    df = pd.DataFrame({
        "title": titles,
        "text": texts
    })

    # When previous step of the pipeline fails to get the title
    # replace the title with the empty string
    df = df.replace({
        "null": ""
    })

    return df

def run_lbl2vec(input_data_path="./data/JSON_outputs/data.json", output_data_path="./data/JSON_outputs/data.json", keywords_csv_path="text_classification/keywords.csv"):
    '''
    input_data_path: the filepath to the json produced by the previous steps of the pipeline
    output_data_path: the filepath to the json that the final json should be written to (can be the same as input_data_path)
    keywords_csv_path: the filepath to the csv with the keywords for each label
    returns: None
    function that runs the lbl2vec model, reading and writing to the appropriate files
    '''

    # Getting the Keywords
    labels = get_keywords_dataframe(keywords_csv_path)

    # Read in the data from the previous steps of the pipeline
    f = open(input_data_path)
    json_object = json.load(f)
    issues = json_object["issues"]

    df = generate_title_text_df(input_json=json_object)

    # tokenize and tag documents combined title + description for Lbl2Vec training
    df['tagged_docs'] = df.apply(lambda row: TaggedDocument(tokenize(row['title'] + '. ' + row['text']), [str(row.name)]), axis=1)

    # init model with parameters
    lbl2vec_model = Lbl2Vec(keywords_list=list(labels['keywords']), tagged_documents=df['tagged_docs'], label_names=list(labels['Subjects']))

    # run the model
    lbl2vec_model.fit()

    # get the similarities from the model
    model_docs_lbl_similarities = lbl2vec_model.predict_model_docs()

    # Get top three most similar labels for each row
    Subjects = list(labels["Subjects"])
    top3_df = model_docs_lbl_similarities[Subjects].apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=3)
    model_docs_lbl_similarities["top3"] = top3_df.apply(lambda row: list(row), axis=1)

    # Iterate through the JSON, and write the predictions to the JSON object
    for issue_number, issue in enumerate(issues):
        articles = issue["articles"]
        for article_number, article in enumerate(articles):
            subjects = model_docs_lbl_similarities["top3"][article_number]
            issues[issue_number]["articles"][article_number]["subjects"] = subjects

    with open(output_data_path, "w") as f:
        json.dump(json_object, f, indent=4)