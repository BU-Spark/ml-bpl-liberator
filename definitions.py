import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input_images")
DEBUG_OUTPUT = os.path.join(DATA_DIR, "output_images")
JSON_OUTPUT = os.path.join(DATA_DIR, "JSON_outputs")
NPY_OUTPUT = os.path.join(DATA_DIR, "npy_outputs")
CROPPED_ARTICLES = os.path.join(DATA_DIR, "cropped_articles")
