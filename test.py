# Required moduls
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from mtcnn.mtcnn import MTCNN
from numpy.core.numeric import NaN
from numpy.lib.function_base import copy
import random

detector = MTCNN()


def reproduce_skin(image, size):
    colors = []

    for y in range(size[1]):
        for x in range(size[0]):
            if not np.array_equal(image[y][x], [0, 0, 0]):
                colors.append(image[y][x])

    for y in range(size[1]):
        for x in range(size[0]):
            if np.array_equal(image[y][x], [0, 0, 0]):
                image[y][x] = colors[random.randrange(len(colors))]

    return image


def crop(img, startx, starty, cropx, cropy):
    return img[starty:starty+cropy, startx:startx+cropx]


def image_dominant_color(a):
    a2D = a.reshape(-1, a.shape[-1])
    a2D = np.delete(a2D, np.where(a2D == [0, 0, 0]), axis=0)
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


sourceImage = cv2.imread("sample/sample3.png", cv2.IMREAD_COLOR)

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([255, 173, 127], np.uint8)


face_detection = []

face_detection = detector.detect_faces(sourceImage)

if len(face_detection) > 0:
    face_box = face_detection[0]['box']
    sourceImage = crop(
        sourceImage, face_box[0], face_box[1], face_box[2], face_box[3])

    h, w = sourceImage.shape[0:2]

    skin_color_final = sourceImage

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skin_color_masked_ycrcb = cv2.bitwise_and(
        imageYCrCb, imageYCrCb, mask=skinRegion)

    avarege_color = np.array(image_dominant_color(skin_color_masked_ycrcb))

    avarege_color_min = np.array(
        avarege_color - avarege_color*0.08, np.uint8)
    avarege_color_max = np.array(
        avarege_color + avarege_color*0.08, np.uint8)

    if avarege_color_max[0] < avarege_color_min[0]:
        temp = avarege_color_max[0]
        avarege_color_max[0] = avarege_color_min[0]
        avarege_color_min[0] = temp

    if avarege_color_max[1] < avarege_color_min[1]:
        temp = avarege_color_max[1]
        avarege_color_max[1] = avarege_color_min[1]
        avarege_color_min[1] = temp

    if avarege_color_max[2] < avarege_color_min[2]:
        temp = avarege_color_max[2]
        avarege_color_max[2] = avarege_color_min[2]
        avarege_color_min[2] = temp

    print(avarege_color)
    print(avarege_color_min)
    print(avarege_color_max)

    skinRegionFix = cv2.inRange(
        imageYCrCb, avarege_color_min, avarege_color_max)

    final_mask = skinRegionFix
    for y in range(h):
        for x in range(w):
            if(skinRegion[y][x] > 0 and skinRegionFix[y][x] > 0):
                final_mask[y][x] = skinRegion[y][x]
            else:
                final_mask[y][x] = 0

    skin_color_final = cv2.bitwise_and(
        sourceImage, sourceImage, mask=final_mask)

    cv2.imwrite('sample/skin-temp.png', reproduce_skin(
        skin_color_final, (w, h)))

    image = Image.open('sample/skin-temp.png')
    image = image.filter(ImageFilter.GaussianBlur(12))
    image = image.save('outrealyFinal2.png')

else:
    raise Exception("Face not found")
