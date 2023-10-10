"""
Manages the processing of where the codes are, and how the image should be positioned to fit between them
"""

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# Start the video stream
vs = VideoStream(src=0).start()

screenspace_corners = [(0, 0), (0, 0), (0, 0), (0, 0)]
default_full_codes = [screenspace_corners for _ in range(4)]

# College maybe?
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
# aruco_params = cv2.aruco.DetectorParameters_create()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()

def get_current_frame():
    """Gets the current frame from the webcam"""
    # Get the video stream from the webcam
    frame = vs.read()
    # frame = imutils.resize(frame, width=1000)

    # Increase the brightness
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    return frame


def vector_from(p1, p2):
    """Calculates the vector from p1 to p2"""
    return [p2[0] - p1[0], p2[1] - p1[1]]


def get_screenspace_points(frame, video_frame, debug, previous_full_codes) -> list[tuple[int, int]]:
    """Gets the points of the codes in the screenspace"""
    # Detect markers in the frame (Aruco 5x5 1000 0-3)
    new_frame = frame.copy()
    # Convert to greyscale
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.threshold(new_frame, 127, 255, cv2.THRESH_BINARY)[1]
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    confirmed = []  # Valid markers visible in the frame

    full_codes = [x.copy() for x in default_full_codes]
    stylus_on = None
    stylus_corners = []
    stage = "Calibration"

    for listID, corner in enumerate(corners[:6]):
        for point in corner:  # for each corner in the list of markers
            # Get the id of the marker
            marker_id = ids[listID][0]
            if marker_id in range(4, 5 + 1):
                for x, y in point:
                    stylus_corners.append((x, y))
                stylus_on = marker_id == 5
            elif marker_id < 4:
                index = 0
                for x, y in point:
                    full_codes[marker_id][index] = (x, y)
                    if debug:
                        # Highlight it if debug is enabled
                        cv2.circle(video_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    # If the point matches the position of the code
                    if index == ids[listID][0] and (x, y) != (0.0, 0.0):
                        # E.G. If the marker is in the top left, this checks for the top left corner of the marker
                        screenspace_corners[index] = (x, y)  # Save it for future frames (used if the marker is lost)
                        confirmed.append(index)
                    index += 1
    if len(confirmed) == 3:
        stage = "Correcting"
    elif len(confirmed) == 4:
        stage = "Accurate"
    # if len(confirmed) == 3 and previous_full_codes != default_full_codes and not debug:
    #     (unknownPoint,) = {0, 1, 2, 3} - set(confirmed)
    #     # Create a matrix that transforms the previous points to the current points
    #     transform_matrix = cv2.getAffineTransform(
    #         np.float32(
    #             [previous_full_codes[confirmed[0]][confirmed[0]],
    #              previous_full_codes[confirmed[1]][confirmed[1]],
    #              previous_full_codes[confirmed[2]][confirmed[2]]]
    #         ), np.float32(
    #             [full_codes[confirmed[0]][confirmed[0]],
    #              full_codes[confirmed[1]][confirmed[1]],
    #              full_codes[confirmed[2]][confirmed[2]]]
    #         )
    #     )
    #     # The previous full code is an array of coordinates, so we need to multiply the matrix by each coordinate
    #     new = []
    #     for point in previous_full_codes[unknownPoint]:
    #         new.append(list(transform_matrix @ np.array([point[0], point[1], 1])))
    #         # Use matmul instead
    #         new.append([max(round(n, 1), 0) for n in list(transform_matrix @ np.array([point[0], point[1], 1]))])
    #     screenspace_corners[unknownPoint] = new[unknownPoint]

    return screenspace_corners, video_frame, full_codes, stylus_corners, stylus_on, stage


def add_screenspace_overlay(frame, screenspace_corners, debug=False):
    """Adds an overlay to the frame showing the screenspace"""
    if (0, 0) not in screenspace_corners:
        screen_corners = np.array([
            [screenspace_corners[0][0], screenspace_corners[0][1]],
            [screenspace_corners[1][0], screenspace_corners[1][1]],
            [screenspace_corners[2][0], screenspace_corners[2][1]],
            [screenspace_corners[3][0], screenspace_corners[3][1]]
        ], dtype="float32")

    if debug:
        for corner in screenspace_corners:
            cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)  # Highlight screen corners

        # Show a full polygon between the 4 corners
        cv2.line(frame, (int(screenspace_corners[0][0]), int(screenspace_corners[0][1])),
                 (int(screenspace_corners[1][0]), int(screenspace_corners[1][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspace_corners[1][0]), int(screenspace_corners[1][1])),
                 (int(screenspace_corners[2][0]), int(screenspace_corners[2][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspace_corners[2][0]), int(screenspace_corners[2][1])),
                 (int(screenspace_corners[3][0]), int(screenspace_corners[3][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspace_corners[3][0]), int(screenspace_corners[3][1])),
                 (int(screenspace_corners[0][0]), int(screenspace_corners[0][1])), (0, 255, 0), 2)

        # Show the diagonals
        cv2.line(frame, (int(screenspace_corners[0][0]), int(screenspace_corners[0][1])),
                 (int(screenspace_corners[2][0]), int(screenspace_corners[2][1])), (0, 255, 0), 2)
        cv2.line(frame, (int(screenspace_corners[1][0]), int(screenspace_corners[1][1])),
                 (int(screenspace_corners[3][0]), int(screenspace_corners[3][1])), (0, 255, 0), 2)
    return frame


def get_midpoints(screenspace_corners, frame, debug):
    """Gets the midpoints of the screenspace corners"""
    midpoints = (  # midpoint of 0&1, 1&2, 2&3, 3&0
        (int((screenspace_corners[0][0] + screenspace_corners[1][0]) / 2),
         int((screenspace_corners[0][1] + screenspace_corners[1][1]) / 2)),
        (int((screenspace_corners[1][0] + screenspace_corners[2][0]) / 2),
         int((screenspace_corners[1][1] + screenspace_corners[2][1]) / 2)),
        (int((screenspace_corners[2][0] + screenspace_corners[3][0]) / 2),
         int((screenspace_corners[2][1] + screenspace_corners[3][1]) / 2)),
        (int((screenspace_corners[3][0] + screenspace_corners[0][0]) / 2),
         int((screenspace_corners[3][1] + screenspace_corners[0][1]) / 2))
    )
    if debug:
        for point in midpoints:
            cv2.circle(frame, point, 5, (255, 0, 255), -1)
        # Show the diagonals
        cv2.line(frame, midpoints[0], midpoints[2], (255, 0, 255), 2)
        cv2.line(frame, midpoints[1], midpoints[3], (255, 0, 255), 2)
    return midpoints, frame


def kill():
    """Stops the video stream gracefully"""
    vs.stop()
