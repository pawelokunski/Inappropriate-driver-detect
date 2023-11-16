from src.estimation_face import get_head_pose_points
from src.detecting_face import load_face_detection_model, detect_faces_in_image
from src.landmarks_face import load_landmark_model, detect_landmarks
from src.eye_tracking import trackbar_callback, display_eye_position, preprocess_threshold, find_contour_and_position, create_eye_mask
import cv2 as cv
import numpy as np
import math
from ultralytics import YOLO


if __name__ == "__main__":
    face_model = load_face_detection_model()
    landmark_model = load_landmark_model()
    left_eye_landmarks = [36, 37, 38, 39, 40, 41]
    right_eye_landmarks = [42, 43, 44, 45, 46, 47]
    phone_detection_model = YOLO('phones.pt')
    hands_detection_model = YOLO('hands.pt')

    cap = cv.VideoCapture(0)
    ret, img = cap.read()
    thresh = img.copy()
    size = img.shape
    font = cv.FONT_HERSHEY_SIMPLEX

    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)])
    cv.namedWindow('image')
    kernel = np.ones((9, 9), np.uint8)

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    cv.createTrackbar('threshold', 'image', 75, 255, trackbar_callback)
    while True:
        ret, img = cap.read()
        yolo = img.copy()
        eyes_img = img.copy()

        if ret:
            faces = detect_faces_in_image(img, face_model)
            for face in faces:
                marks = detect_landmarks(img, landmark_model, face)
                image_points = np.array([marks[30], marks[8], marks[36],
                                                marks[45], marks[48], marks[54]], dtype="double")
                dist_coeffs = np.zeros((4, 1))
                (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs, flags=cv.SOLVEPNP_UPNP)

                (nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = get_head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv.line(img, p1, p2, (0, 255, 255), 2)
                cv.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                try:
                    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90

                try:
                    m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1 / m)))
                except:
                    ang2 = 90

                if ang1 >= 48:
                    print('Head down')
                    cv.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                elif ang1 <= -48:
                    print('Head up')
                    cv.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

                if ang2 >= 48:
                    print('Head right_eye_landmarks')
                    cv.putText(img, 'Head right_eye_landmarks', (90, 30), font, 2, (255, 255, 128), 3)
                elif ang2 <= -48:
                    print('Head left_eye_landmarks')
                    cv.putText(img, 'Head left_eye_landmarks', (90, 30), font, 2, (255, 255, 128), 3)

                cv.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                cv.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
            img2 = img
            cv.imshow('img', img2)

            rects = detect_faces_in_image(eyes_img, face_model)

            for rect in rects:
                shape = detect_landmarks(eyes_img, landmark_model, rect)
                mask = np.zeros(eyes_img.shape[:2], dtype=np.uint8)
                mask, end_points_left = create_eye_mask(mask, left_eye_landmarks, shape)
                mask, end_points_right = create_eye_mask(mask, right_eye_landmarks, shape)
                mask = cv.dilate(mask, kernel, 5)

                eyes = cv.bitwise_and(eyes_img, eyes_img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv.cvtColor(eyes, cv.COLOR_BGR2GRAY)
                threshold = cv.getTrackbarPos('threshold', 'image')
                _, thresh = cv.threshold(eyes_gray, threshold, 255, cv.THRESH_BINARY)
                thresh = preprocess_threshold(thresh)

                eyeball_pos_left = find_contour_and_position(thresh[:, 0:mid], mid, eyes_img, end_points_left)
                eyeball_pos_right = find_contour_and_position(thresh[:, mid:], mid, eyes_img, end_points_right, True)
                display_eye_position(eyes_img, eyeball_pos_left, eyeball_pos_right)

            cv.imshow('eyes', eyes_img)

            results = phone_detection_model(yolo, verbose=False)
            print("Phone detected") if len(results[0].boxes.cls) > 0 else None
            annotated_frame = results[0].plot()
            cv.imshow("YOLOv8 Inference2", annotated_frame)

            results2 = hands_detection_model(yolo, verbose=False)
            annotated_frame2 = results2[0].plot()
            cv.imshow("YOLOv8 Inference", annotated_frame2)
            print("Hands not on wheel") if ((len(results2[0].boxes.cls.tolist()) > 0
                                            and 3. not in results2[0].boxes.cls.tolist())
                                            or len(results2[0].boxes.cls.tolist()) == 0) else None

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv.destroyAllWindows()
    cap.release()
