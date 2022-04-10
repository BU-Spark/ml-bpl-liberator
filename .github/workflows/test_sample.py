import definitions
import argparse
import os
import sys
from colorama import Fore, init
init(autoreset=True)
from argparse import ArgumentParser

from ocr import column_extractor_scc
from ocr.extract_articles.extract_polygons import segment_all_images
sys.path.append('./bbz-segment/05_prediction/src')
import detect_gen
from ocr import crop_ocr
from ner.stanza_ner import StanzaNER

test_directory = "./test_sample/8k71pf94q"
if test_directory is not None:
    if not os.path.exists(test_directory):
        print(Fore.RED + "Invalid input directory given: " + test_directory)
        print(Fore.RED + "Quitting...")
        sys.exit()
    else:
        definitions.INPUT_DIR = test_directory


print(Fore.CYAN + "Running column extraction...")
os.makedirs(definitions.SEGMENT_OUTPUT, exist_ok=True)
column_extractor_scc.run_columns(definitions.INPUT_DIR, definitions.SEGMENT_OUTPUT)

print(Fore.CYAN + "Running object detection model...")
detect_gen.bulk_generate_separators(definitions.INPUT_DIR, 'jpg', definitions.NPY_OUTPUT, True, False, False)

print(Fore.CYAN + "Running article segmentation...")
for item in os.listdir(definitions.INPUT_DIR):
    folder = os.path.join(definitions.INPUT_DIR, item)
    if os.path.isdir(folder):
        npy_out = os.path.join(definitions.NPY_OUTPUT, item)
        debug_out = os.path.join(definitions.DEBUG_OUTPUT, item)
        os.makedirs(npy_out, exist_ok=True)
        os.makedirs(debug_out, exist_ok=True)
        segment_all_images(npy_out, folder, definitions.SEGMENT_OUTPUT, item+"_segment.json", False)

if not os.path.exists(os.path.join(definitions.CONFIG_DIR, "credentials.json")):
    print(Fore.RED + "No Google Cloud credentials found, unable to complete OCR/NER")
    print(Fore.RED + "Quitting...")
    sys.exit()

print(Fore.CYAN + "Loading NER Models...")
NER_PIPELINE = StanzaNER()

print(Fore.CYAN + "Producing OCR & NER for issues...")
for item in os.listdir(definitions.SEGMENT_OUTPUT):
    if not item.startswith('cols') and not item.startswith('.DS_Store'):
        crop_ocr.issue_ocr(os.path.join(definitions.SEGMENT_OUTPUT, item), NER_PIPELINE)

print(Fore.GREEN + "Complete, wrote data.json to " + definitions.JSON_OUTPUT)
