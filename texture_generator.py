# face color analysis given eye center position
import argparse

import cv2
import numpy as np
import random
from PIL import Image, ImageFilter, ImageOps
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


# to detect the eyes
eyes = cv2.CascadeClassifier("eye2.xml")
faces = cv2.CascadeClassifier("face.xml")


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

    pixels = np.float32(a.reshape(-1, 3))

    n_colors = 8
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dic = dict()
    for i in range(len(palette)):
        dic[counts[i]] = palette[i]

        np.sort(counts)
    dominant = dic[counts[1]]

    simple_array = []
    for i in range(len(counts)):
        color = dic[counts[i]]
        if(color > 10).all():
            simple_array.append(color)

    # return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    return simple_array


def average_colour(image):
    colour_tuple = [None, None, None]
    for channel in range(3):

        # Get data for one channel at a time
        pixels = image.getdata(band=channel)

        values = []
        for pixel in pixels:
            values.append(pixel)

        colour_tuple[channel] = sum(values) / len(values)

    return tuple(colour_tuple)


def imageOpen(name):
    return Image.open(f"eye{name}.png")


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='sample/sample2.png',
                    help="face face_image")
parser.add_argument('--input_path_uv', default='sample/sample2-uv.png',
                    help="uv map created from face_image")
parser.add_argument('--out_dir', default='out/',
                    help="output directory")
opt = parser.parse_args()

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray",
              "Brown Black", "Green", "Green Gray", "Green Hazel", "Other")
EyeColor = {
    class_name[0]: ((156, 21, 50), (255, 100, 85)),
    class_name[1]: ((146, 2, 10), (320, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((50, 21, 50), (155, 100, 85)),
    class_name[6]: ((50, 2, 25), (145, 20, 65)),
    class_name[7]: ((25, 20, 17), (60, 45, 55)),

}


def check_color(hsv_input, color):
    hsv = hsv_input
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and \
            (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and \
            (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    else:
        return False


# define eye color category rules in HSV space

def convert_hsv_to_percent(color):
    cvH = color[0]/180*360
    cvS = (color[1] / 255) * 100
    cvV = (color[2] / 255) * 100
    return cvH, cvS, cvV


def find_class(hsv):
    color_id = len(class_name)-1
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id


def find_eye_position(face_image):
    imgHSV = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
    h, w = face_image.shape[0:2]
    imgMask = np.zeros((face_image.shape[0], face_image.shape[1], 1))

    result = detector.detect_faces(face_image)
    if result == []:
        raise Exception("The face was NOT found!")
        return

    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    return(left_eye)


def eye_color(image):
    # now the face is in the frame
    # the detection is done with the gray scale frame
    left_eye = find_eye_position(image)
    imgcopy = image.copy()
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))

    # eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = 500 / 30  # approximate

    print(left_eye)
    # print(right_eye)

    cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
    # cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 2)
    # cv2.circle(face_image, right_eye, int(eye_radius), (0, 155, 255), 2)

    eye_class = np.zeros(len(class_name), np.float)

    # version 1
    # kernal = np.ones((5, 5), "uint8")

    # for i in range(len(class_name) - 1):

    #     print(f"TEST :  {EyeColor2[class_name[i]][0]}")
    #     green_lower = np.array(EyeColor2[class_name[i]][0], np.uint8)
    #     green_upper = np.array(EyeColor2[class_name[i]][1], np.uint8)
    #     green_mask = cv2.inRange(imgHSV, green_lower, green_upper)
    #     green_mask = cv2.dilate(green_mask, kernal)
    #     res_green = cv2.bitwise_and(imgcopy, imgcopy, mask=green_mask)

    #     contours, hierarchy = cv2.findContours(green_mask,
    #                                            cv2.RETR_TREE,
    #                                            cv2.CHAIN_APPROX_SIMPLE)
    #     for pic, contour in enumerate(contours):
    #         area = cv2.contourArea(contour)
    #         if(area > 300):
    #             x, y, w, h = cv2.boundingRect(contour)
    #             imgcopy = cv2.rectangle(imgcopy, (x, y),
    #                                     (x + w, y + h),
    #                                     (0, 255, 0), 1)

    #             cv2.putText(imgcopy, f"{class_name[i]} Colour", (x, y),
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.4, (0, 255, 0))

    #     cv2.imshow("Multiple Color Detection in Real-TIme", imgcopy)
    #     cv2.waitKey(0)

    # Version 2
    # clasify each pixel with mask where is eye
    firstpoint = 0
    for y in range(0, h):
        for x in range(0, w):
            if imgMask[y, x] != 0:
                # debug %12 point
                converted_color = convert_hsv_to_percent(imgHSV[y, x])
                if firstpoint % 8 == 0:

                    print(
                        f"RGB{converted_color} - {class_name[find_class(converted_color)]}")
                firstpoint += 1
                eye_class[find_class(converted_color)] += 1

    cv2.imwrite('sample/hsv-eye.jpg', imgHSV)
    cv2.imwrite('sample/mask.jpg', imgMask)

    # check color which is used the most
    main_color_index = np.argmax(eye_class[:len(eye_class) - 1])
    total_vote = eye_class[:len(eye_class) - 1].sum()

    print("\n\nDominant Eye Color: ", class_name[main_color_index])
    print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i] / total_vote, 2))

    # open all images
    eyes_images = []
    for i in range(len(class_name) - 1):
        eyes_images.append(imageOpen(i + 1))

    eye_image = eyes_images[main_color_index].copy()

    print(f"\nMAIN: {main_color_index}\n")

    # this will be blending few other colros to the main, by the % of rate
    for i in range(len(class_name)):
        value = 0
        if(total_vote > 0):
            value = round(eye_class[i] / total_vote, 2)

        if i != main_color_index:
            print(f"{class_name[i]} - alpha: {value}")
            # eye_image = Image.blend(eyes_images[i], eye_image, value)

        else:
            print(f"{class_name[i]} - alpha: {value} <---")

        # mask = Image.new("L", eyes_images[0].size, value)

    eye_image.save(opt.out_dir + "eye-final.png")


def generate_hair_texture(uv_face_texture):
    # create hair by cuted image from  UVtexture
    img_hair_color = uv_face_texture.copy()
    img_hair_color = img_hair_color.crop((250, 25, 256, 40))
    mask = Image.new("L", img_hair_color.size, 128)
    img_hair_color2 = uv_face_texture.crop((0, 25, 6, 40))

    img_hair_color = Image.composite(img_hair_color, img_hair_color2, mask)
    img_hair_color = img_hair_color.resize((256, 256))
    img_hair_color.save(opt.out_dir + "hair-final.png")


def generate_skin_texture(uv_face_texture):
    # create skin by color taken from face UVtexture
    img_face = uv_face_texture.copy()
    img_background = Image.open('head-background.png').convert('L')
    img_face_color = img_face.copy()
    img_alpha = Image.open('alpha.png').convert('L')

    img_face_color = img_face_color.crop((124, 54, 132, 60))
    im_a_blur = img_face_color.filter(ImageFilter.GaussianBlur(40))
    # pobieram rozmiary
    bx, by = img_background.size

    ix, iy = img_face.size
    img_face = img_face.resize((int(ix * 1.6), int(iy * 1.6)))
    ix, iy = img_face.size

    # wysrodkowywuje i podnosze obrazek
    coordinate_x = int(bx / 2 - ix / 2)
    coordinate_y = int(by / 2 - iy / 2 - 170)

    img_background = ImageOps.colorize(
        img_background, (0, 0, 0), average_colour(im_a_blur))

    img_alpha = img_alpha.resize(img_face.size)

    # wstawiam twarz w tło z maską
    img_background.paste(img_face, (coordinate_x, coordinate_y), img_alpha)
    # img_background = img_background.resize((bx*2, by*2))
    img_background.save(opt.out_dir + "head-final.png")


def mask_by_color(image, color):
    avarege_color_min = np.array(
        color - color*0.09, np.uint8)
    avarege_color_max = np.array(
        color + color*0.09, np.uint8)

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

    print(color)
    print(avarege_color_min)
    print(avarege_color_max)

    skinRegionFix = cv2.inRange(
        image, avarege_color_min, avarege_color_max)

    return skinRegionFix


def generate_skin_texture2(face_texture_path, uv_face_texture):

    sourceImage = cv2.imread(face_texture_path, cv2.IMREAD_COLOR)

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

        cv2.imwrite('sample/debug-skin.png', skinRegion)

        skin_color_masked_ycrcb = cv2.bitwise_and(
            imageYCrCb, imageYCrCb, mask=skinRegion)
        cv2.imwrite('sample/debug-skin2.png', skin_color_masked_ycrcb)

        avarege_colors = np.array(
            image_dominant_color(skin_color_masked_ycrcb))

        print(f"colors: {avarege_colors}")

        final_mask = np.zeros((h, w), np.uint8)

        for i in range(2):

            current_mask = mask_by_color(imageYCrCb, avarege_colors[i])

            for y in range(h):
                for x in range(w):
                    if(skinRegion[y][x] > 0 and (final_mask[y][x] > 0 or current_mask[y][x] > 0)):
                        final_mask[y][x] = skinRegion[y][x]
                    else:
                        final_mask[y][x] = 0

        skin_color_final = cv2.bitwise_and(
            sourceImage, sourceImage, mask=final_mask)

        cv2.imwrite('sample/debug-median-skin.png', skin_color_final)
        cv2.imwrite('sample/skin-temp.png', reproduce_skin(
            skin_color_final, (w, h)))

        skin_reproduced = Image.open('sample/skin-temp.png')
        skin_reproduced = skin_reproduced.filter(ImageFilter.GaussianBlur(32))

        img_uv_face = uv_face_texture.copy()
        img_background = skin_reproduced.copy()
        img_background = img_background.resize((1024, 1024))

        img_face_color = img_uv_face.copy()
        img_alpha = Image.open('alpha.png').convert('L')

        img_face_color = img_face_color.crop((124, 54, 132, 60))
        im_a_blur = img_face_color.filter(ImageFilter.GaussianBlur(40))

        # pobieram rozmiary
        bx, by = img_background.size

        ix, iy = img_uv_face.size
        img_uv_face = img_uv_face.resize((int(ix * 1.6), int(iy * 1.6)))
        ix, iy = img_uv_face.size

        # wysrodkowywuje i podnosze obrazek
        coordinate_x = int(bx / 2 - ix / 2)
        coordinate_y = int(by / 2 - iy / 2 - 170)

        img_alpha = img_alpha.resize(img_uv_face.size)

        # wstawiam twarz w tło z maską
        img_background.paste(
            img_uv_face, (coordinate_x, coordinate_y), img_alpha)
        # img_background = img_background.resize((bx*2, by*2))
        img_background.save(opt.out_dir + "head-final.png")

    else:
        raise Exception("Face not found")


if __name__ == '__main__':
    image = cv2.imread(opt.input_path, cv2.IMREAD_COLOR)
    eye_color(image)
    cv2.imwrite('sample/result.jpg', image)

    # twarz
    img_face_uv_input = Image.open(opt.input_path_uv)

    generate_skin_texture2(opt.input_path, img_face_uv_input)
    generate_hair_texture(img_face_uv_input)
