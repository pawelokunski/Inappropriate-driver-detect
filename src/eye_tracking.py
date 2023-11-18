import cv2 as cv
import numpy as np


def create_eye_mask(mask, eye_side, facial_landmarks):
    eye_points = [facial_landmarks[i] for i in eye_side]
    eye_points = np.array(eye_points, dtype=np.int32)
    mask = cv.fillConvexPoly(mask, eye_points, 255)
    left = eye_points[0][0]
    top = (eye_points[1][1] + eye_points[2][1]) // 2
    right = eye_points[3][0]
    bottom = (eye_points[4][1] + eye_points[5][1]) // 2
    return mask, [left, top, right, bottom]


def calculate_eyeball_position(end_points, center_x, center_y):
    x_ratio = (end_points[0] - center_x) / (center_x - end_points[2])
    y_ratio = (center_y - end_points[1]) / (end_points[3] - center_y)
    if x_ratio > 2.3:
        return 1  # Left
    elif x_ratio < 0.33:
        return 2  # Right
    elif y_ratio < 0.33:
        return 3  # Up
    else:
        return 0  # Center


def find_contour_and_position(threshold, mid, image, end_points, is_right=False):
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    try:
        contour = max(contours, key=cv.contourArea)
        moments = cv.moments(contour)
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])
        if is_right:
            center_x += mid
        cv.circle(image, (center_x, center_y), 4, (0, 0, 255), 2)
        position = calculate_eyeball_position(end_points, center_x, center_y)
        return position
    except:
        pass


def preprocess_threshold(threshold):
    threshold = cv.erode(threshold, None, iterations=2)
    threshold = cv.dilate(threshold, None, iterations=4)
    threshold = cv.medianBlur(threshold, 3)
    threshold = cv.bitwise_not(threshold)
    return threshold


def display_eye_position(image, left_position, right_position):
    if left_position == right_position and left_position != 0:
        text = ''
        if left_position == 1:
            print('Looking left')
            text = 'Looking left'
        elif left_position == 2:
            print('Looking right')
            text = 'Looking right'
        elif left_position == 3:
            print('Looking up')
            text = 'Looking up'
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, text, (30, 30), font,
                   1, (0, 255, 255), 2, cv.LINE_AA)


def trackbar_callback(x):
    pass
