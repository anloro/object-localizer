import logging
from object_detector import ObjectDetector
from object_localizer import ObjectLocalizer
from aircraft import Aircraft

from utils.utils import getImgpaths
import cv2 as cv

def main():

    uav = Aircraft()
    detector = ObjectDetector(uav)

    img_paths = getImgpaths("assets")
    flag = True
    while(flag):
        # frame = uav.get_next_frame()
        try:
            img = img_paths.popitem()[1]
            frame = cv.imread(img)
            detector.detect_object(frame)
        except KeyError:
            flag = False
            print("End of dataset")


if __name__ == "__main__":
    main()
