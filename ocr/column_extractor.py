
# https://github.com/jscancella/NYTribuneOCRExperiments/blob/master/findText_usingSums.py
import os
import io
from pathlib import Path
import sys
os.environ['OPENCV_IO_ENABLE_JASPER']='True' # has to be set before importing cv2 otherwise it won't read the variable
import numpy as np
import cv2

import subprocess
from multiprocessing import Pool
from scipy.signal import find_peaks, find_peaks_cwt

import scipy.ndimage as ndimage
from IPython.display import Image as KImage

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


# Run adaptative thresholding (is slow af compared to not using it in pipeline)
def adaptative_thresholding(img, threshold):
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
CREATE_COLUMN_OUTLINE_IMAGES = True # if we detect that we didn't find all the columns. Create a debug image (tiff) showing the columns that were found

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

def createColumnImages(img, basename, directory):
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
    
    peaks, _ = find_peaks(sums, distance=400) # the column indexs of the img array, spaced at least 800 away from the previous peak


    sum_to_index = dict((sums[peaks[i]], peaks[i]) for i in range(len(peaks)))
    sorted_sums = sorted(sum_to_index.keys())
    '''
    qr = Q_test(sorted_sums)
    if qr:
        peaks = peaks[peaks != sum_to_index[sorted_sums[0]]]
    '''
    print("PeakNum, Sum, QRemove for " + basename)
    for x in peaks:
        print(str(x) + ', ' + str(sums[x]))
    print("----------")

    if peaks.size == 0:
        with open('troublesomeImages.txt', 'a') as f:
            print("ERROR: something went wrong with finding the peaks for image: ", os.path.join(directory, basename))
            f.write(os.path.join(directory, basename) + ".jpg 0\n")
        return files

    peaks[0] = 0 # automatically make the left most column index the start of the image
    peaks[-1] =sums.size -1 # automatically make the right most column index the end of the image

    boxed = np.copy(img)
    if peaks.size < 6:
        with open('troublesomeImages.txt', 'a') as f:
            print("found image that is causing problems: ", os.path.join(directory, basename))
            f.write(os.path.join(directory, basename) + ".jpg " + str(peaks.size) + "\n")

    columnIndexPairs = columnIndexes(peaks)

    ystart = 0
    yend = img.shape[0]
    for columnIndexPair in columnIndexPairs:
        xstart = max(columnIndexPair[0]-PADDING, 0)
        xend = min(columnIndexPair[1]+PADDING, img.shape[1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, '%s_xStart%s_xEnd%s.jpg' % (basename, xstart,xend))
        files.append(filepath)
        crop_img = img[ystart:yend, xstart:xend]
        
        print("writing out cropped image: ", filepath)
        # Apply adaptative thresholding to the image with a threshold of 25/100
        #crop_img = adaptative_thresholding(crop_img, 25)
        if not cv2.imwrite(filepath, crop_img):
            print('failed')

        if CREATE_COLUMN_OUTLINE_IMAGES:
            cv2.rectangle(boxed,(xstart,ystart),(xend,yend), GREEN, LINE_THICKNESS)

    if CREATE_COLUMN_OUTLINE_IMAGES:
        filepath = os.path.join(directory, '%s-contours.jpeg' % basename)
        cv2.imwrite(filepath, boxed, [cv2.IMWRITE_JPEG_QUALITY, 50])
        # For removing the old image?
        # os.remove(os.path.join(directory, basename + ".jp2"))

    return files

def invert_experiment():
    test_img = cv2.imread('./ocr/data/8k71pf94q/1_commonwealth_8k71pf94q_accessFull.jpg')
    for thresh in range(1, 200, 20):
        print('writing thresh= ' + str(thresh))
        _,temp_img = cv2.threshold(test_img, thresh, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('./ocr/test_images/thresh='+str(thresh)+'.jpg', temp_img)



def test(img, basename):
    #h, w, _ = img.shape
    #test_img = cv2.imread('./ocr/data/8k71pf94q/2_commonwealth_8k71pf94q_accessFull.jpg')
    test_img = convertToGrayscale(img)
    #ret,test_img = cv2.threshold(test_img,25,255,0)
    #cv2.imwrite('./ocr/test_images/contours/'+basename+'prepixelcrop.jpg', test_img)
    #test_img = test_img[10:h-10, 10: w-10]
    #y_nonzero, x_nonzero = np.nonzero(test_img)
    #test_img = test_img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
    test_img = invert(test_img)
    test_img = dilateDirection(test_img)

    #contours,hierarchy = cv2.findContours(test_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #x,y,w,h = cv2.boundingRect(cnt)
    #test_img = cv2.rectangle(img,(10,10),(w-10, h-10), GREEN, LINE_THICKNESS)
    #test_img = cv2.drawContours(test_img, contours, -1, GREEN, LINE_THICKNESS)
    #crop = test_img[y:y+h,x:x+w]
    cv2.imwrite('./ocr/test_images/contours/'+basename+'dilated.jpg', test_img)
    '''
    for r in range(0, 40, 5):
        name = 'rank=' + str(r) + ".jpg"
        path = './ocr/test_images/' + name

        new_img = ndimage.rank_filter(test_img, rank=r, size=20)
        print("writing " + name)
        cv2.imwrite(path, new_img)
    '''
    #cv2.imwrite('./ocr/test_images/inverted.jpg', test_img)

    


if __name__ == "__main__":
    print("STARTING")
    for f in os.listdir('./ocr/data/gb19gw39h/'):
        if f.endswith(".jpg"):
            #test(cv2.imread(os.path.join('./ocr/data/gb19gw39h/', f)), 'gb19gw39h-' + f[0])
            createColumnImages(cv2.imread(os.path.join('./ocr/data/gb19gw39h/', f)), 'gb19gw39h-' + f[0], './ocr/columns/gb19gw39h/')

    for f in os.listdir('./ocr/data/8k71pf94q/'):
        if f.endswith(".jpg"):
            #test(cv2.imread(os.path.join('./ocr/data/gb19gw39h/', f)), 'gb19gw39h-' + f[0])
            createColumnImages(cv2.imread(os.path.join('./ocr/data/8k71pf94q/', f)), '8k71pf94q-' + f[0], './ocr/columns/8k71pf94q/')

    for f in os.listdir('./ocr/data/mc87rq85m/'):
        if f.endswith(".jpg"):
            #test(cv2.imread(os.path.join('./ocr/data/gb19gw39h/', f)), 'gb19gw39h-' + f[0])
            createColumnImages(cv2.imread(os.path.join('./ocr/data/mc87rq85m/', f)), 'mc87rq85m-' + f[0], './ocr/columns/mc87rq85m/')

    '''
    data_folder = './ocr/data/'
    for folder in os.listdir(data_folder):
        if folder == ".DS_Store":
            continue
        for file in os.listdir(os.path.join(data_folder, folder)):
            if file.endswith(".jpg"):
                print("calling test() on " + file)
                #test(cv2.imread(os.path.join(data_folder, folder, file)),folder+'-'+file[0])
                createColumnImages(cv2.imread(os.path.join(data_folder, folder, file)), folder+'-'+file[0], './ocr/columns/'+folder+'/')
    
    for f in os.listdir('./ocr/data/8k71pr786/'):
        if f.endswith(".jpg"):
            for d in range(550, 850, 50):
                createColumnImages(cv2.imread(os.path.join('./ocr/data/8k71pr786/', f)), '8k71pr786-'+f[0]+'-d=' + str(d), './ocr/test_images/test_contour/8k71pr786/', d)
            #createColumnImages(cv2.imread('./ocr/data/8k71pr786/'), 'tester2', './ocr/data/columns/tester/')
    '''

