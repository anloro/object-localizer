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
        #Set the white mask
        white_mask = self.white_filter(frame)
        white_mask = cv.cvtColor(white_mask, cv.COLOR_BGR2GRAY)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv.dilate(white_mask, kernel, iterations=2)
        
        #Set the threshold 
        thresh = self.threshold(frame)

        #Join both methods 
        thresh_mask = cv.bitwise_and(thresh, thresh, mask=white_mask)    
        
        kernel_dilate= np.ones((5, 5), np.uint8)
        filtered_image = cv.dilate(thresh_mask, kernel_dilate, iterations=2)

        kernel_erode= np.ones((3, 3), np.uint8)
        filtered_image = cv.erode(filtered_image, kernel_erode, iterations=1)
        
        #Take the edge 
        frame_result = self.edge_filter(filtered_image)

        
        f, axarr = plt.subplots(1,3, figsize=(50,25))
        axarr[0].imshow(white_mask)
        axarr[1].imshow(thresh)
        axarr[2].imshow(frame_result)
        plt.waitforbuttonpress(0)
        plt.close()


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

        # Set the centroide of the contours
        for i in squares:
            M = cv.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv.circle(frame, (cx, cy), 2, (0, 0, 255), -1) 

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

        frame_boxed = cv.drawContours(frame, positions, -1, (0, 255, 0), 2)
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
        pure_white_hsv = np.array([0, 0, 0])
        pale_gray_hsv = np.array([360, 60, 255])

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

    def threshold(self, frame):
        """Create a threshold filter using its own function.
        input:
            - frame: Input image
        output:
            - frame_result: Filtered image.
        """
        # Convert BGR to HSV
        imgray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(imgray,(5,5),0)
        _, frame_threshold = cv.threshold(blur, 200 ,255,cv.THRESH_BINARY)
        frame_result = cv.cvtColor(frame_threshold, cv.COLOR_GRAY2BGR)

        return frame_result


    def edge_filter(self, frame):
        """Use a edge filter to preprocess the image and enhance contour detection."""
        gausBlur = cv.GaussianBlur(frame, (5,5),0)
        filtered_image = cv.Canny(gausBlur, 0, 200)

        kernel_dil= np.ones((5, 5), np.uint8)
        filtered_image = cv.dilate(filtered_image, kernel_dil, iterations=1)
        kernel_er= np.ones((3, 3), np.uint8)
        filtered_image = cv.erode(filtered_image, kernel_er, iterations=1)
        

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