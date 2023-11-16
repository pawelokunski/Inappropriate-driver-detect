import cv2 as cv
import numpy as np


def load_face_detection_model():
    model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "models/deploy.txt"
    face_model = cv.dnn.readNetFromCaffe(config_file, model_file)
    return face_model


def detect_faces_in_image(image, face_model):
    height, width = image.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    result = face_model.forward()
    detected_faces = []
    for i in range(result.shape[2]):
        confidence = result[0, 0, i, 2]
        if confidence > 0.5:
            box = result[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            detected_faces.append([x, y, x1, y1])
    return detected_faces
