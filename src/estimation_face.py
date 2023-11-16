import cv2 as cv
import numpy as np


def calculate_2d_points(image, rotation_vector, translation_vector, camera_matrix, values):
    points_3d = []
    distortion_coefficients = np.zeros((4, 1))

    rear_size = values[0]
    rear_depth = values[1]
    points_3d.append((-rear_size, -rear_size, rear_depth))
    points_3d.append((-rear_size, rear_size, rear_depth))
    points_3d.append((rear_size, rear_size, rear_depth))
    points_3d.append((rear_size, -rear_size, rear_depth))
    points_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = values[2]
    front_depth = values[3]
    points_3d.append((-front_size, -front_size, front_depth))
    points_3d.append((-front_size, front_size, front_depth))
    points_3d.append((front_size, front_size, front_depth))
    points_3d.append((front_size, -front_size, front_depth))
    points_3d.append((-front_size, -front_size, front_depth))

    points_3d = np.array(points_3d, dtype=np.float).reshape(-1, 3)

    (points_2d, _) = cv.projectPoints(points_3d,
                                       rotation_vector,
                                       translation_vector,
                                       camera_matrix,
                                       distortion_coefficients)
    points_2d = np.int32(points_2d.reshape(-1, 2))

    return points_2d


def get_head_pose_points(image, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = image.shape[1]
    front_depth = front_size * 2
    values = [rear_size, rear_depth, front_size, front_depth]

    points_2d = calculate_2d_points(image, rotation_vector, translation_vector, camera_matrix, values)
    y = (points_2d[5] + points_2d[8]) // 2
    x = points_2d[2]

    return (x, y)
