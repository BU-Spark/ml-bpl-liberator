import stanza

def ner(ocr_text):
    nlp = stanza.Pipeline('en')
    doc = nlp(ocr_text)
    print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
    return doc


    