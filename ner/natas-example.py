# https://github.com/mikahama/natas
import natas

print("\n[INFO] CACHE is enabled by default for natas")

print("Loading spacy model...")

def correct_raw_ocr(ocr_text, nlp):

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

