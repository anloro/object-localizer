import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class ObjectDetector:
    """Algorithms used for the object detection."""

    def __init__(self, aircraft):
        self.aircraft = aircraft
        pass

    def preprocess_image(self, frame):
        """Filters the image only leaving certain tones f
        of white.

        input:
            - frame (np.array): Image that needs pre-processing.
        output:
            - frame_result (np.array): Preprocessed image.
        """

        white_mask = self.white_filter(frame)
        white_mask = cv.cvtColor(white_mask, cv.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv.dilate(white_mask, kernel, iterations=2)
    
        edge_mask = self.edge_filter(frame)

        frame_result = cv.bitwise_and(edge_mask, edge_mask, mask=white_mask)

        f, axarr = plt.subplots(1,3, figsize=(16,8))
        axarr[0].imshow(white_mask)
        axarr[1].imshow(edge_mask)
        axarr[2].imshow(frame_result)
        plt.show() 

        # cv.imshow("frame_result", frame_result)
        # cv.waitKey(0)

        return frame_result

    def detect_object(self, frame):
        """Routine that gets a frame, preprocesses it
        detects the contours and plots them.

        input:
            -
        output:
            -
        """
        processed_frame = self.preprocess_image(frame)

        # Change frame to gray for the findContours algorithm
        contours, _ = cv.findContours(processed_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Process the contours
        squares = self.approximate_squares(contours)

        self.draw_bounding_box(squares, frame)

    def draw_bounding_box(self, positions, frame):
        """Sets the bounding box of the detected object.

        input:
            - position (np.array): Position of
            the upper left corner of the bounding box.
            - frame (np.array): Image that needs the bounding box.
        output:
            - frame_boxed (np.array): image with the bounding box.
        """

        frame_boxed = cv.drawContours(frame, positions, -1, (0, 255, 0), 3)
        cv.imshow("frame_boxed", frame_boxed)
        cv.waitKey(0)

        return frame_boxed

    def white_filter(self, frame):
        """Creates a mask to filter the image using a white threshold.

        input:
            - frame: Input image
        output:
            - frame_result: Filtered image.
        """
        # Convert BGR to HSV
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Threshold of white in HSV space
        pure_white_hsv = np.array([0, 0, 80])
        pale_gray_hsv = np.array([255, 60, 255])

        # Create and use the mask
        mask = cv.inRange(frame_hsv, pure_white_hsv, pale_gray_hsv)
        # Get rid off noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=2)
        mask = cv.dilate(mask, kernel, iterations=2)

        # Apply the mask
        frame_masked_hsv = cv.bitwise_and(frame_hsv, frame_hsv, mask=mask)
        # Return to BGR format
        frame_result = cv.cvtColor(frame_masked_hsv, cv.COLOR_HSV2BGR)

        return frame_result

    def edge_filter(self, img):
        """Use a edge filter to preprocess the image and enhance contour detection."""
        gausBlur = cv.GaussianBlur(img, (5,5),0)
        filtered_image = cv.Canny(gausBlur, 100, 200)

        kernel = np.ones((5, 5), np.uint8)
        filtered_image = cv.dilate(filtered_image, kernel, iterations=1)

        return filtered_image

    def approximate_squares(self, contours):
        """
        Computes the arc length and approximates a polygon.
        If the approximated polygon has 4 vertexes it saves the
        results and it returns it.
        """
        squares = list()
        for contour in contours:
            arc_length = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.04 * arc_length, True)
            if len(approx) == 4:
                squares.append(approx)

        return squares
