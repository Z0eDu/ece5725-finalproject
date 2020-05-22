import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
import glob
import os

# save numpy array as npy file
from numpy import asarray
from numpy import save



def tag_extraction(image):
    # read image and apply color filter
    img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    mask = cv2.inRange(hsv, (30, 40, 20), (90, 255,255))
    
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

    # detect circles
    detected_circles = cv2.HoughCircles(gray,  
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                   param2 = 1, minRadius = 10, maxRadius = 50) 

    if detected_circles is not None: 

        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles))
        #print(detected_circles)
        pt = detected_circles[0,0]
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(green, (a, b), r, (255, 255, 255), 2) 

    # obtain roi
    roi = img[b-r:b+r,a-r:a+r]
    
    # return empty if on the border
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        tag = np.zeros((78,78))
        tag = np.uint8(tag)
        return tag
    
    # make roi into a square if not already
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if gray_roi.shape[0] != gray_roi.shape[1]:
        if gray_roi.shape[0] < gray_roi.shape[1]:
            new_roi = np.zeros((gray_roi.shape[1],gray_roi.shape[1]))
            new_roi[0:gray_roi.shape[0],0:gray_roi.shape[1]] = gray_roi
        else: 
            new_roi = np.zeros((gray_roi.shape[0],gray_roi.shape[0]))
            new_roi[0:gray_roi.shape[0],0:gray_roi.shape[1]] = gray_roi
        gray_roi = new_roi

    # crop or append to 78*78
    if gray_roi.shape[0] > 78:
        gray_roi = gray_roi[r-39:r+39,r-39:r+39]
        tag = gray_roi
    elif roi.shape[0] < 78:
        tag = np.zeros((78,78))
        tag[39-r:39+r,39-r:39+r] = gray_roi
        tag = np.uint8(tag)
    else:
        tag = gray_roi
    
    return tag


# insert your own directory path here
dirpath = ['/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/0',
       '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/0_1',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/1',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/2',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/3',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/4',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/5',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/6',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/7',
      '/Users/DeanWang/Desktop/Spring_2020/ECE5725/LAB/Final/TAG/8']


# store training image dataset and label set
for i,foldername in enumerate (dirpath):
    for filename in glob.glob(foldername+'/*.jpg'):
        print(filename)
        image = cv2.imread(filename)
        tag = tag_extraction(image)
        tag = tag.reshape(1,tag.shape[0]*tag.shape[1])
        train.append(tag)
        if i == 0 or i == 1:
            label.append([0])
        else:
            label.append([i-1])



train = asarray(train)
label = asarray(label)

# save to a npy file
save('train.npy', train)
save('label.npy', label)