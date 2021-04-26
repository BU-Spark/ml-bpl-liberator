import stanza
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import definitions

class StanzaNER:

    '''
    This class exists only to run the run_ner method. This is put in a class so that the project's main.py only has to load the Stanza models once.
    '''

    def __init__(self):
        self.nlp = stanza.Pipeline('en', dir=definitions.STANZA_RESOURCES_DIR)
        self.tags = {"GPE", "LOC", "PERSON", "NORP", "ORG"}


    def run_ner(self, ocr_text):
        
        doc = self.nlp(ocr_text)

        ner_dict = {}

        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.type in self.tags:
                    if ent.type in ner_dict:
                        if not ent.text in ner_dict[ent.type]:
                            ner_dict[ent.type].append(ent.text)
                    else:
                        ner_dict[ent.type] = [ent.text]

        return ner_dict

