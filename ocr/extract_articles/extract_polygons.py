import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from colorama import Fore
import numpy as np
import cv2 as cv

from ocr.extract_articles.labeled_page import LabeledPage
from ocr.extract_articles.data_classes import Article
from ocr.extract_articles.polygon_utils import random_color


def segment_all_images(saved_labels_dir: str, orig_img_dir: str, output_folder: str, json_file_name: str, debug=False) -> int:
    """
    Top Level function to handle running the segmentation code against a folder of images.
    Args:
        saved_labels_dir (): The location of the .npy files
        orig_img_dir (): The location of original images that correspond to the .npy file
        output_folder (): The folder of where to save output files(can be same as input)
        debug (): Enables debug mode

    Returns:
        The number of articles identified

    """
    if not Path(saved_labels_dir).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(output_folder).exists():
        raise FileNotFoundError("Folder not found")
    if not Path(orig_img_dir).exists():
        raise FileNotFoundError("Folder not found")

    numpy_file_list = list(Path(saved_labels_dir).glob('*.npy'))
    master_article_list: List[Article] = list()
    file_skip_list: List[str] = list()
    for file in tqdm(numpy_file_list):
        try:
            name = str(file.stem).split('_')
            issue_id = name[2]
            image_id = name[0]
            #1_commonwealth_8k71pf94q_accessFull.sep
        except ValueError:
            print(f'{Fore.LIGHTYELLOW_EX}\n skipped file: {str(file)}')
            print(f'could not identify issue and image ID in file name, skipping{Fore.RESET}')
            file_skip_list.append(str(file))
            continue

        loaded_data = np.load(str(file), allow_pickle=True)
        labels, original_size, filename = loaded_data.item().get('labels'), loaded_data.item().get(
            'dimensions'), loaded_data.item().get('filename')

        if labels is None or original_size is None or filename is None:
            print(f'{Fore.RED} \n failed to load labels from file {str(file)}. Skipped.')
            print(f'{Fore.RESET}')
            file_skip_list.append(str(file))
            continue
        # Must swap around, the size is given in a tuple of (height, width)
        # While the LabeledImage wants (width, height)
        original_size = original_size[1], original_size[0]
        print('filename= ' + str(filename))
        articles = segment_single_image(labels, original_size, issue_id, image_id,
                                        src_img_path=str(Path(orig_img_dir).joinpath(filename)), debug=debug,
                                        output_folder=output_folder)
        master_article_list.extend(articles)

    # Dump data to json
    with open(Path(output_folder).joinpath(json_file_name), 'w') as outfile:
        print('Opening JSON file at: ' + str(Path(output_folder).joinpath(json_file_name)))
        json.dump([val.JSON() for val in master_article_list], outfile, indent=4)

    return len(master_article_list)


def segment_single_image(input_img_array: np.array, original_size: Tuple[int, int], issue_id: str = None,
                         img_id: str = None, src_img_path: str = None,
                         debug=False, output_folder: str = None) \
        -> List[Article]:
    """
    Top level function to segment a single image
    Args:
        input_img_array (): The numpy array loaded from .npy file that contains labels from the ML model
        original_size (): The original image size, in the form of (height, width)
        issue_id (): The ID for this issue
        img_id (): The ID of this image
        src_img_path (): The path to where the original image is stored
        debug (): Enable debug mode
        output_folder (): Folder to save output files/

    Returns:

        The list of identified articles.

    """
    if debug:
        print("checking: " + str(Path(src_img_path)) + " exists?: " + str(Path(src_img_path).exists()))
        if not Path(src_img_path).exists():
            raise FileNotFoundError
        if not Path(output_folder).exists():
            raise FileNotFoundError

    page = LabeledPage(input_img_array, original_size, img_id, issue_id, str(Path(src_img_path).name))
    # Takes a bit to run
    articles = page.segment_single_image()
    if debug:
        annotate_image(src_img_path, articles, output_folder)
        '''
        try:
            annotate_image(src_img_path, articles, output_folder)
        except Exception:
            print(
                f'{Fore.LIGHTYELLOW_EX}\nFailed to load source image for annotation. '
                f'\n Annotated image not saved. \n IMG: {src_img_path} {Fore.RESET}')
        '''
    return articles


def annotate_image(input_img_scr: str, articles: List[Article], output_folder: str) -> None:
    """
    This function is used for debugging. It annotates an image with colored boxes used to represent
    identified regions of articles on the image.
    Args:
        input_img_scr (): Path to the original image
        articles (): List of articles to draw on image
        output_folder (): Location of where to save annotated image

    Returns: Nothing

    """
    source_img = cv.imread(input_img_scr)
    source_img_name = Path(input_img_scr).stem
    for index, article in enumerate(articles):
        color = random_color()
        for box in article.img_boxes:
            # Inspired by the following:
            # https://gist.github.com/jdhao/1cb4c8f6561fbdb87859ac28a84b0201
            rect = cv.minAreaRect(box.get_contours())
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(source_img, [box], 0, color, 5)
    cv.imwrite(str(Path(output_folder).joinpath(f'{source_img_name}_annotated.jpg')), source_img)
