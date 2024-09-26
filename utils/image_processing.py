import cv2
import numpy as np

def return_GRAY2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def gaussian_noise(img, maxValue, block_size, c):
    data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gaussian_img = cv2.adaptiveThreshold(data, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)

    return return_GRAY2RGB(gaussian_img)


def canny(img, th1, th2):
    data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(data, th1, th2)

    return return_GRAY2RGB(edges)

def sharpening(img, median): 
    data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.zeros((3, 3)) - 1
    f[1][1] = median

    sharpening = cv2.filter2D(data, -1, f)

    return return_GRAY2RGB(sharpening)

def threshold_otsu(img): 
    data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # data = cv2.GaussianBlur(data, (0, 0), 1)
    data = cv2.Canny(data, 100, 100)
    t, otsu_img = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return return_GRAY2RGB(otsu_img)

def gaussian_blur(img, sigma_value, kernal_size = (0, 0)):
    data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gaussian_blur_img = cv2.GaussianBlur(data, kernal_size, sigma_value)

    return return_GRAY2RGB(gaussian_blur_img)