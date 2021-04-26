from flask import Flask, request, jsonify
import spacy
import natas
from autocorrect import Speller

nlp = spacy.load('en_core_web_md')
spell = Speller()

app = Flask(__name__)

@app.route('/natas_correct', methods=['POST'])
def ocr_correct():

    incorrect = []
    words = []
    
    # get the input text form-data encoded 
    input_ocr_text = request.form['text']
    print(input_ocr_text)
    
    output = input_ocr_text
    
    doc = nlp(input_ocr_text)
 
    for w in doc:
        word = w.text
        correct = natas.is_correctly_spelled(word)
        if not correct:
            incorrect.append(word)
        words.append(word)

    candidates = natas.ocr_correct_words(incorrect)

    for w,c in zip(incorrect, candidates):
        # Retrieve the top candidate and perform replacement
        if len(c) != 0:
            print(w + "\t" + c[0])
            output = output.replace(w, c[0])
    
    # return the response containing both the original and corrected text
    response = {"original": input_ocr_text, "corrected": output}

    return response
    

@app.route('/autocorrect', methods=['POST'])
def autocorrect():
    input_ocr_text = request.form['text']
    only_replacements = bool(request.form['only_replacements'])
    if only_replacements:
        spell2 = Speller(only_replacements=True)
        corrected = spell2(input_ocr_text)
    else:
        corrected = spell(input_ocr_text)
    
    response = {"original": input_ocr_text, "corrected": corrected}

    return response


if __name__ == '__main__':
    app.run()
    

