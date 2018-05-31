import math

import cv2
import numpy as np
import random
from collections import deque

cap = cv2.VideoCapture(1)
# To keep track of all point where object visited
center_points = deque()

while True:
    # Read and flip frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Blur the frame a little
    blur_frame = cv2.GaussianBlur(frame, (7, 7), 0)

    # Convert from BGR to HSV color format
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper range of hsv color to detect. Blue here
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Make elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Opening morph(erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find all contours
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) > 0:
        # Find the biggest contour
        biggest_contour = max(contours, key=cv2.contourArea)

        # Find center of contour and draw filled circle
        moments = cv2.moments(biggest_contour)
        centre_of_contour = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        cv2.circle(frame, centre_of_contour, 5, (0, 0, 255), -1)

        # Bound the contour with circle
        ellipse = cv2.fitEllipse(biggest_contour)
        cv2.ellipse(frame, ellipse, (0, 255, 255), 2)

        # Save the center of contour so we draw line tracking it
        center_points.appendleft(centre_of_contour)

    # Draw line from center points of contour
    for i in range(1, len(center_points)):
        b = random.randint(230, 255)
        g = random.randint(100, 255)
        r = random.randint(100, 255)
        if math.sqrt(((center_points[i - 1][0] - center_points[i][0]) ** 2) + (
                (center_points[i - 1][1] - center_points[i][1]) ** 2)) <= 50:
            cv2.line(frame, center_points[i - 1], center_points[i], (b, g, r), 4)

    cv2.imshow('original', frame)
    cv2.imshow('mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
