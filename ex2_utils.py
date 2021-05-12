"""
        '########:'##::::'##:::########:::::
         ##.....::. ##::'##::###:::;###:::
         ##::::::::. ##'##::::.. ::##::::
         ######:::::. ###:::::::  ##:
         ##...:::::: ## ##::::::##::
         ##:::::::: ##:. ##::::##::
         ########: ##:::. ##::########::
        ........::..:::::..:::......::
"""
from typing import List

import cv2
import numpy as np
from collections import defaultdict

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315090167

def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    kenrnelLength = len(kernel1)
    inSignal= np.pad(inSignal, (kenrnelLength-1,kenrnelLength-1), )
    signalLength= len(inSignal)
    convArray= np.zeros(signalLength - kenrnelLength + 1)
    for i in range(signalLength - kenrnelLength +1):
        convArray[i]= (inSignal[i:i+len(kernel1)] * kernel1).sum()
    return convArray

def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    kernel2= np.flip(kernel2)
    heightKer, widthKer = kernel2.shape
    heightImg, widthImg = inImage.shape
    image_padded = np.pad(inImage, ( heightKer // 2, widthKer // 2), 'edge')
    convImg = np.zeros((heightImg, widthImg))
    for i in range(heightImg):
        for j in range(widthImg):
            convImg[i,j] = (image_padded[i:i + heightKer, j:j + widthKer] * kernel2).sum()
    return convImg

def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    kernel1= np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernel2=kernel1.transpose()
    x_der= conv2D(inImage, kernel1)
    y_der= conv2D(inImage, kernel2)
    directrions= np.arctan(y_der, x_der)
    mangitude = np.sqrt(np.square(x_der)+ np.square(y_der))

    return directrions , mangitude , x_der, y_der


def createGaussianKer(kernel_size, sigma):
    center=(int)(kernel_size/2)
    kernel=np.zeros((kernel_size,kernel_size))
    for i in range(kernel_size):
       for j in range(kernel_size):
          diff=np.sqrt((i-center)**2+(j-center)**2)
          kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    kernel=createGaussianKer(kernel_size, sigma)
    return conv2D(in_image,kernel)

def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    sigma = int(round(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8))
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    edgeCV = cv2.magnitude(x, y)
    edgeCV[edgeCV < thresh*255] = 0
    edgeCV[edgeCV >= thresh*255] = 1

    #my implementation - follow https://en.wikipedia.org/wiki/Sobel_operator
    gradValX= conv2D(img, Gx)
    gradValY=conv2D(img, Gy)
    magnitude= np.sqrt(np.square(gradValX)+np.square(gradValY))
    magnitude[magnitude < thresh * 255] = 0
    magnitude[magnitude >= thresh*255] = 1

    return edgeCV, magnitude
def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using the "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    width= img.shape[1]
    height= img.shape[0]
    lapKernel= np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    #Smooth with 2D Gaussian
    img = conv2D(img, lapKernel)
    logImage= np.zeros(img.shape)
    for i in range(height-(lapKernel.shape[0]-1)):
        for j in range(width-(lapKernel.shape[1]-1)):
            if img[i, j] == 0:
                #check all neighbors
                if (img[i - 1][j] < 0 and img[i + 1][j] > 0) or \
                        (img[i - 1][j] > 0 and img[i + 1][j] < 0) or\
                        (img[i][j - 1] < 0 and img[i][j + 1] > 0) or \
                        (img[i][j - 1] < 0 and img[i][j + 1] < 0):
                    logImage[i][j] = 255
            if img[i, j]<0:
                if (img[i - 1][j] > 0) or (img[i + 1][j] > 0) or (img[i][j-1] > 0) or (img[i][j+1] > 0):
                    logImage[i][j] = 255
    return logImage

'''
    """
    Quant the degree using the method learned at class
    :param direction: Input  current direction
    :return: direction after quantizaztion
    """
'''
def quantDirections(direction):
    direction[(direction < 22.5) | (157.5 <= direction)] = 0
    direction[(direction < 67.5) | (22.5 <= direction)] = 45
    direction[(direction < 112.5) | (67.5 <= direction)] = 90
    direction[(direction < 157.5) | (112.5 <= direction)] = 135
    return  direction


def nonMaximumSuppression(magnitude, direction):
    ans = np.zeros(magnitude.shape)
    # - For each pixel (x,y) compare to pixels along its gradient direction.
    # - If |G(x,y)| is not a local maximum, set it to zero
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            if direction[i, j] == 0:
                if (magnitude[i, j] > magnitude[i, j - 1]) and (magnitude[i, j] > magnitude[i, j + 1]):
                    ans[i, j] = magnitude[i, j]
            elif direction[i, j] == 45:
                if (magnitude[i, j] > magnitude[i - 1, j + 1]) and (magnitude[i, j] > magnitude[i + 1, j - 1]):
                    ans[i, j] = magnitude[i, j]
            elif direction[i, j] == 90:
                if (magnitude[i, j] > magnitude[i - 1, j]) and (magnitude[i, j] > magnitude[i + 1, j]):
                    ans[i, j] = magnitude[i, j]
            elif direction[i, j] == 135:
                if (magnitude[i, j] > magnitude[i - 1, j - 1]) and (magnitude[i, j] > magnitude[i + 1, j + 1]):
                    ans[i, j] = magnitude[i, j]
    return ans

'''function to check if a given pos in the img is connected to edge'''
def connectToEdge(result, i, j):
    if result[i-1, j-1] == 255 or result[i - 1, j - 1] == 255 or result[i - 1, j]== 255 or result[i - 1, j + 1]== 255 or result[i, j - 1] == 255 or result[i, j + 1] ==  255 or result[i + 1, j - 1]== 255 or result[i + 1, j]==255 or result[i + 1, j + 1]==255 : return True
    return False


def hysteresis(img, thrs_1, thrs_2,  initial = False):
    '''
     Define two thresholds T1 > T2
- Every pixel with |G(x,y)| greater than T1 is presumed
to be an edge pixel.
- Every pixel which is both
(1) connected to an edge pixel, and
(2) has |G(x,y)| greater than T2
is also selected as an edge pixel.
    :param img:
    :param thrs_1:
    :param thrs_2:
    :return: edge detection pic
     '''
    low=thrs_1*255
    high= thrs_2*255
    print("Low", low, "High", high)

    img_h, img_w = img.shape
    result = np.zeros((img_h, img_w))
    result[img >= thrs_1] = 255
    weak = np.argwhere((img <= thrs_1) & (img >= thrs_2))
    for i, j in weak:
        result[i, j]= 255 if connectToEdge(result, i, j) else 0
    result[img< thrs_2]= 0
    return result

def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float)-> (np.ndarray, np.ndarray):
    """
    Detecting edges using2 "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """
    #cv implemenatation
    edgeImgCv2 = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), thrs_1 * 255, thrs_2 * 255)
    # my implementation:
    # smooth the image
    img= cv2.GaussianBlur(img, (5, 5), 0)
    # compute partial derivatives
    gradValX = conv2D(img, Gx)
    gradValY = conv2D(img, Gy)
    # compute magnitude & direction
    magnitude= np.sqrt((np.square(gradValX)+np.square(gradValY)))
    direction=np.rad2deg(np.arctan2(gradValX, gradValY)) % 180
    #  Quantize the gradient directions:
    direction= quantDirections(direction)
    # Perform non-maximum suppression
    edc =nonMaximumSuppression(magnitude, direction)
    #  Hysteresis
    edgeImg= hysteresis(edc , thrs_1, thrs_2)
    return edgeImgCv2, edgeImg
def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    # use this sorce: https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.py
    #  Use the Canny Edge detector as the edge detector
    height, width = img.shape
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 50, 100)
    # detect all edges
    edges = np.argwhere(img > 0)
    # Initializing accumulator array.
    # Accumulator array is a 3 dimensional array with the dimensions representing
    # the radius, X  and Y coordinate .
    # Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((max_radius, height + 2 * max_radius, width + 2 * max_radius))
    # Precomputing all angles
    theta = np.arange(0, 360) * np.pi / 180
    for r in range(round(min_radius), round(max_radius)):
        # Creating a Circle Blueprint
        bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
        (m, n) = (r + 1, r + 1)  #center of blueprint
        for angle in theta:
            x = int(np.round(r * np.cos(angle)))
            y = int(np.round(r * np.sin(angle)))
            bprint[m + x, n + y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x, y in edges:  # For each edge coordinates
            # Centering the blueprint circle over the edges
            # and updating the accumulator array
            # X = [x - m + max_radius, x + m + max_radius]  # Computing the extreme X values
            # Y = [y - n + max_radius, y + n + max_radius]  # Computing the extreme Y values
            A[r, x-m +max_radius: x+m+max_radius, y-n + max_radius: y+n+max_radius] += bprint
        threshold=7
        A[r][A[r] < threshold * constant / r] = 0
    # size to detect peaks
    region =15
    B = np.zeros((max_radius, height + 2 * max_radius, width + 2 * max_radius))
    #extracting the  circles detected
    for r, x, y in np.argwhere(A):
        temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
        p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
        B[r + (p - region), x + (a - region), y + (b - region)] = 1
    circles= np.argwhere(B[:, max_radius:-max_radius, max_radius:-max_radius])
    return circles


