
# https://github.com/jscancella/NYTribuneOCRExperiments/blob/master/findText_usingSums.py
import os
import io
import sys
os.environ['OPENCV_IO_ENABLE_JASPER']='True' # has to be set before importing cv2 otherwise it won't read the variable
import numpy as np
import cv2
import random
from scipy.signal import find_peaks
import scipy.ndimage as ndimage
from IPython.display import Image as KImage
import json

#custom kernel that is used to blend together text in the Y axis
DILATE_KERNEL = np.array([
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)


# Run adaptative thresholding (is slow compared to not using it in pipeline)
def adaptative_thresholding(img, threshold):
    '''
    Unused and not necessary for effectiveness of column extraction, full B/W thresholding works best.
    '''
    # Load image
    I = img
    # Convert image to grayscale
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # Original image size
    orignrows, origncols = gray.shape
    # Windows size
    M = int(np.floor(orignrows/16) + 1)
    N = int(np.floor(origncols/16) + 1)
    # Image border padding related to windows size
    Mextend = round(M/2)-1
    Nextend = round(N/2)-1
    # Padding image
    aux =cv2.copyMakeBorder(gray, top=Mextend, bottom=Mextend, left=Nextend,
                          right=Nextend, borderType=cv2.BORDER_REFLECT)
    windows = np.zeros((M,N),np.int32)
    # Image integral calculation
    imageIntegral = cv2.integral(aux, windows,-1)
    # Integral image size
    nrows, ncols = imageIntegral.shape
    # Memory allocation for cumulative region image
    result = np.zeros((orignrows, origncols))
    # Image cumulative pixels in windows size calculation
    for i in range(nrows-M):
        for j in range(ncols-N):
            result[i, j] = imageIntegral[i+M, j+N] - imageIntegral[i, j+N]+ imageIntegral[i, j] - imageIntegral[i+M,j]

    # Output binary image memory allocation
    binar = np.ones((orignrows, origncols), dtype=np.bool)
    # Gray image weighted by windows size
    graymult = (gray).astype('float64')*M*N
    # Output image binarization
    binar[graymult <= result*(100.0 - threshold)/100.0] = False
    # binary image to UINT8 conversion
    binar = (255*binar).astype(np.uint8)

    return binar
    

def Q_test(sorted_data):
    '''
    Dixon's Q-Test to remove false positive in column separators. Unused.
    '''
    conf95_level = {3: .97, 4: .829, 5: .71, 6: .625, 7: .568, 8: .526, 9: .493}
    q_exp = abs(sorted_data[1] - sorted_data[0]) / abs(sorted_data[-1] - sorted_data[0])
    print(str(abs(sorted_data[1] - sorted_data[0])) + ' / ' + str(abs(sorted_data[-1] - sorted_data[0])))
    print("q_exp : " + str(q_exp))
    return q_exp > conf95_level[min(9, len(sorted_data))]


# static variables for clarity
COLUMNS = 0
GREEN = (0, 255, 0)

# parameters that can be tweaked
LINE_THICKNESS = 3 # how thick to make the line around the found contours in the debug output
PADDING = 10 # padding to add around the found possible column to help account for image skew and such
CREATE_CROPPED_IMAGES = False # Creates and outputs the actual cropping based on contour lines

def columnIndexes(a):
    """
    creates pair of indexes for left and right index of the image column
    For example [13, 1257, 2474, 3695, 4907, 6149]
    becomes: [[13 1257], [1257 2474], [2474 3695], [3695 4907], [4907 6149]]
    """
    nrows = (a.size-2)+1
    return a[1*np.arange(nrows)[:,None] + np.arange(2)]

def convertToGrayscale(img):
    temp_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return temp_img

def invert(img):
    """ Black -> White | White -> Black """
    print("invert image")
    # Should we edit these parameters?
    #3/18/21 - experimented on threshold, 140 is good.
    _,temp_img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)
    return temp_img

def dilateDirection(img, debug=False):
    """
    It is just opposite of erosion. Here, a pixel element is '1' if atleast one pixel under the kernel is '1'. 
    So it increases the white region in the image or size of foreground object increases. 
    Normally, in cases like noise removal, erosion is followed by dilation. 
    Because, erosion removes white noises, but it also shrinks our object. 
    So we dilate it. Since noise is gone, they won't come back, but our object area increases. 
    It is also useful in joining broken parts of an object. 
    """
    print("applying dilation morph")
    temp_img = cv2.dilate(img, DILATE_KERNEL, iterations=15) #the more iterations the more the text gets stretched in the Y axis, 15 seems about right.
    '''
    if debug:
        filepath = os.path.join(debugOutputDirectory, '%s-dilation.tiff' % basename)
        cv2.imwrite(filepath, temp_img)
    '''
    return temp_img

def createColumnImages(img, basename, directory, debug=False):
    """
    we sum each column of the inverted image. The columns should show up as peaks in the sums
    uses scipy.signal.find_peaks to find those peaks and use them as column indexes
    """

    files = []
    temp_img = convertToGrayscale(img)
    temp_img = invert(temp_img)
    temp_img = dilateDirection(temp_img)
    
    sums = np.sum(temp_img, axis = COLUMNS)
    
    sums[0] = 1000 # some random value so that find_peaks properly detects the peak for the left most column
    sums = sums * -4 # invert so that minimums become maximums and exagerate the data so it is more clear what the peaks are
    if basename.startswith("gb"):
        dist = 400
    else:
        dist = 600
    peaks, _ = find_peaks(sums, distance=dist) # the column indexs of the img array, spaced at least 800 away from the previous peak

    if peaks.size < 5 or peaks.size > 7:
        with open('troublesomeImages.txt', 'a') as f:
            print("ERROR: something went wrong with finding the peaks for image: ", os.path.join(directory, basename))
            f.write(os.path.join(directory, basename) + ".jpg 0\n")
        return files

    peaks[0] = 0 # automatically make the left most column index the start of the image
    peaks[-1] =sums.size -1 # automatically make the right most column index the end of the image

    boxed = np.copy(img)
    columnIndexPairs = columnIndexes(peaks)

    if debug:

        if not os.path.exists(directory):
                    os.makedirs(directory)

        ystart = 0
        yend = img.shape[0]
        for columnIndexPair in columnIndexPairs:
            xstart = max(columnIndexPair[0]-PADDING, 0)
            xend = min(columnIndexPair[1]+PADDING, img.shape[1])

            if CREATE_CROPPED_IMAGES:
                filepath = os.path.join(directory, '%s_xStart%s_xEnd%s.jpg' % (basename, xstart,xend))
                files.append(filepath)

            cv2.rectangle(boxed,(xstart,ystart),(xend,yend), GREEN, LINE_THICKNESS)

        filepath = os.path.join(directory, '%s-contours.jpeg' % basename)
        cv2.imwrite(filepath, boxed, [cv2.IMWRITE_JPEG_QUALITY, 50])
        # For removing the old image?
        # os.remove(os.path.join(directory, basename + ".jp2"))

    return columnIndexPairs


def run_columns(input_dir, output_dir, debug=False):

    '''
    Entry point for extracting the column separator indices. 
    Args:

    input_dir (str): directory of input images. Must contain either images directly or subdirectories each containing 
    a subset of images (such as one full issue). If debug is true, output will imitate the subdirectory structure of input_dir.

    output_dir (str): location of output JSON

    debug (bool): if true, outputs images with contour lines dividing columns

    Output: if debug, images with contour lines. If not debug, writes json in output_dir
    '''

    dict_for_json = {}

    for item in os.listdir(input_dir):
        print("working on " + item)
        folder = os.path.join(input_dir, item)
        if os.path.isdir(folder):
            cols_dict = dict()
            for file in os.listdir(folder):
                if file.endswith('.jpg'):
                    print("in file " + file)
                    pairs = createColumnImages(cv2.imread(os.path.join(folder, file)), folder+'-'+file[0], output_dir, debug).tolist()
                    cols_dict[file] = pairs
            dict_for_json[item] = cols_dict
        else:
            if item.endswith('.jpg'):
                pairs = createColumnImages(cv2.imread(os.path.join(input_dir, item)), item, output_dir, debug).tolist()
                dict_for_json[item] = pairs


    with open(os.path.join(output_dir, 'cols.json'), 'a') as out:
        print(dict_for_json)
        json.dump(dict_for_json, out, indent=4)