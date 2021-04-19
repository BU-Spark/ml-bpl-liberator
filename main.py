import definitions
import argparse
import os
import sys

from ocr import column_extractor_scc
from ocr.extract_articles.extract_polygons import segment_all_images
sys.path.append('./bbz-segment/05_prediction/src')
import detect_gen


print("Running column extraction...")
column_extractor_scc.run_columns(definitions.INPUT_DIR, definitions.JSON_OUTPUT)

print("Running object detection model...")
detect_gen.bulk_generate_separators(definitions.INPUT_DIR, 'jpg', definitions.NPY_OUTPUT, True, False, False)

print("Running article segmentation...")
for item in os.listdir(definitions.INPUT_DIR):
    folder = os.path.join(definitions.INPUT_DIR, item)
    if os.path.isdir(folder):
        npy_out = os.path.join(definitions.NPY_OUTPUT, item)
        debug_out = os.path.join(definitions.DEBUG_OUTPUT, item)
        os.makedirs(npy_out, exist_ok=True)
        os.makedirs(debug_out, exist_ok=True)
        segment_all_images(npy_out, folder, definitions.JSON_OUTPUT, item+"_segment.json", False)
