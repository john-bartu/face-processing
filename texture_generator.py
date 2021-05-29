# face color analysis given eye center position
import argparse

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


# to detect the eyes
eyes = cv2.CascadeClassifier("eye2.xml")
faces = cv2.CascadeClassifier("face.xml")


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
parser.add_argument('--input_path', default='sample/sample5.png',
                    help="face face_image")
parser.add_argument('--input_path_uv', default='sample/sample3-uv.png',
                    help="uv map created from face_image")
parser.add_argument('--out_dir', default='out/',
                    help="output directory")
opt = parser.parse_args()

# define HSV color ranges for eyes colors
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray",
              "Brown Black", "Green", "Green Gray" ,"Other")
EyeColor = {
    class_name[0]: ((156, 21, 50), (240, 100, 85)),
    class_name[1]: ((146, 2, 25), (300, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((50, 21, 50), (155, 100, 85)),
    class_name[6]: ((50, 2, 25), (145, 20, 65))
    
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


def find_class(hsv):
    color_id = 7
    for i in range(len(class_name)-1):
        if check_color(hsv, EyeColor[class_name[i]]) == True:
            color_id = i

    return color_id


def find_eye_position(face_image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))

    result = detector.detect_faces(image)
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
                if firstpoint % 8 == 0:
                    print(
                        f"RGB{imgHSV[y, x]} - {class_name[find_class(imgHSV[y, x])]}")
                firstpoint += 1
                eye_class[find_class(imgHSV[y, x])] += 1

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


if __name__ == '__main__':
    image = cv2.imread(opt.input_path, cv2.IMREAD_COLOR)
    eye_color(image)
    cv2.imwrite('sample/result.jpg', image)

    # twarz
    img_face_uv_input = Image.open(opt.input_path_uv)

    generate_skin_texture(img_face_uv_input)
    generate_hair_texture(img_face_uv_input)
