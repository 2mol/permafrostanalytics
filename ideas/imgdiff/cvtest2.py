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

    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    hsv_channels = cv2.split(hsv)

    rows = diff.shape[0]
    cols = diff.shape[1]

    for i in range(0, rows):
        for j in range(0, cols):
            h = hsv_channels[0][i][j]

            if h > 90 and h < 130:
                hsv_channels[2][i][j] = 255
            else:
                hsv_channels[2][i][j] = 0

    # cv2.imshow("show", hsv_channels[0])
    # cv2.imshow("show2", hsv_channels[2])

    # -------------------------------------------

    # cv2.imwrite('diff.png', difference)
    # concat_image = np.concatenate((image1, hsv_channels[0], image2), axis=1)

    # cv2.imshow(image1_name, image1)
    # cv2.imshow(image2_name, image2)

    while True:
        # cv2.imshow('imgdiff', resize(concat_image, 0.25))
        cv2.imshow('imgdiff', hsv_channels[0])

        k = cv2.waitKey(0)
        if k == 27:  # Esc
            break

    cv2.destroyAllWindows()
