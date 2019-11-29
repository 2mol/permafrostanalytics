import numpy as np
import cv2


WIDTH, HEIGHT = 1600, 1000
SCALE = 0.25


def resize(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


if __name__ == '__main__':
    # image1_name = "timelapse_images/2017-09-16/20170916_120007.JPG"
    # image2_name = "timelapse_images/2017-09-17/20170917_120003.JPG"
    image1_name = "20170923_120003.JPG"
    image2_name = "20170925_120004.JPG"
    image1 = cv2.imread(image1_name)
    image2 = cv2.imread(image2_name)

    # window = cv2.namedWindow(image1_name, cv2.WINDOW_NORMAL)
    # window = cv2.namedWindow(image2_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(image1_name, width, height)
    # cv2.resizeWindow(image2_name, width, height)
    window = cv2.namedWindow('imgdiff', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgdiff', WIDTH, HEIGHT)

    # diff_left = cv2.subtract(image1, image2)
    # diff_right = cv2.subtract(image1, image2)
    # diff = cv2.add(diff_left, diff_right)

    # diff = cv2.subtract(image1, image2)
    diff = cv2.absdiff(image1, image2)

    # -------------------------------------------

    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # diff[thresh == 255] = 0

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # diff = cv2.erode(diff, kernel, iterations = 1)

    # -------------------------------------------

    # cv2.imwrite('diff.png', difference)
    concat_image = np.concatenate((image1, diff, image2), axis=1)

    # cv2.imshow(image1_name, image1)
    # cv2.imshow(image2_name, image2)

    while True:
        cv2.imshow('imgdiff', resize(concat_image, 0.25))

        k = cv2.waitKey(0)
        if k == 27:  # Esc
            break

    cv2.destroyAllWindows()
