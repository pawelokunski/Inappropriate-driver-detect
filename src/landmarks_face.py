import cv2
import numpy as np
import tensorflow as tf


def load_landmark_model():
    landmark_model = tf.saved_model.load("models/pose_model")
    return landmark_model


def get_square_bbox(bbox):
    left_x = bbox[0]
    top_y = bbox[1]
    right_x = bbox[2]
    bottom_y = bbox[3]

    bbox_width = right_x - left_x
    bbox_height = bottom_y - top_y

    diff = bbox_height - bbox_width
    delta = int(abs(diff) / 2)

    if diff == 0:
        return bbox
    elif diff > 0:
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    assert ((right_x - left_x) == (bottom_y - top_y)), 'Bounding box is not square.'

    return [left_x, top_y, right_x, bottom_y]


def shift_bbox(bbox, offset):
    left_x = bbox[0] + offset[0]
    top_y = bbox[1] + offset[1]
    right_x = bbox[2] + offset[0]
    bottom_y = bbox[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def detect_landmarks(image, model, face_bbox):
    offset_y = int(abs((face_bbox[3] - face_bbox[1]) * 0.1))
    moved_bbox = shift_bbox(face_bbox, [0, offset_y])
    square_bbox = get_square_bbox(moved_bbox)

    height, width = image.shape[:2]
    if square_bbox[0] < 0:
        square_bbox[0] = 0
    if square_bbox[1] < 0:
        square_bbox[1] = 0
    if square_bbox[2] > width:
        square_bbox[2] = width
    if square_bbox[3] > height:
        square_bbox[3] = height

    face_image = image[square_bbox[1]: square_bbox[3], square_bbox[0]: square_bbox[2]]
    face_image = cv2.resize(face_image, (128, 128))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    predictions = model.signatures["predict"](
        tf.constant([face_image], dtype=tf.uint8))

    landmarks = np.array(predictions['output']).flatten()[:136]
    landmarks = np.reshape(landmarks, (-1, 2))

    landmarks *= (square_bbox[2] - square_bbox[0])
    landmarks[:, 0] += square_bbox[0]
    landmarks[:, 1] += square_bbox[1]
    landmarks = landmarks.astype(np.uint)

    return landmarks


def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    for landmark in landmarks:
        cv2.circle(image, (landmark[0], landmark[1]), 2, color, -1, cv2.LINE_AA)
