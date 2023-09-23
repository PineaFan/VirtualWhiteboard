"""
Manipulates the video feed and creates the matrices for transforming points in and out of screenspace coordinates
"""

import cv2
import numpy as np
import tkinter as tk


class OverlayOptions:
    def __init__(self):
        ...
    STRETCH = 0


def generate_warp_matrix(source_image, new_corners, fit_option: OverlayOptions = OverlayOptions.STRETCH):
    """Generates a matrix from the corners of the screen to the corners of the codes"""
    # Create a matrix which maps the points of sourceImage to the points of newCorners
    image_corners = np.array([
        [0, 0],
        [source_image.shape[1], 0],
        [source_image.shape[1], source_image.shape[0]],
        [0, source_image.shape[0]]
    ], dtype="float32")
    screen_corners = np.array([
        [new_corners[0][0], new_corners[0][1]],
        [new_corners[1][0], new_corners[1][1]],
        [new_corners[2][0], new_corners[2][1]],
        [new_corners[3][0], new_corners[3][1]]
    ], dtype="float32")
    # Create the matrix
    warp_matrix = cv2.getPerspectiveTransform(image_corners, screen_corners)
    return warp_matrix


def find_new_coordinate(point, warp_matrix):
    """Finds the new position of a point after it has been warped"""
    # Applies the warp matrix to a point
    # By multiplying the point by the matrix, we can find the new position of the point
    try:
        point = np.array([point[0], point[1], 1])
        new_point = np.matmul(warp_matrix, point)
        if new_point[2] != 0:  # Prevent division by 0
            new_point = new_point / new_point[2]
        return new_point[0], new_point[1]
    except Exception as e:
        print(e)
        return point[:2]


def warp_image(image, warp_matrix, dimensions, fit_option: OverlayOptions = OverlayOptions.STRETCH):
    """Takes an image and a warp matrix and returns the image warped to the new position"""
    return cv2.warpPerspective(image, warp_matrix, (dimensions[1], dimensions[0]))


def overlay_image(base, overlay, warp_matrix, fit_option: OverlayOptions = OverlayOptions.STRETCH):
    """Adds an image to another image"""
    warped_image = warp_image(overlay, warp_matrix, base.shape[:2])
    # Generate a mask by making every pixel is not transparent pure black
    # This does not work with translucent images
    mask = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Apply the mask to the background image
    base = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask))
    # Then add the overlay to the background
    base = cv2.addWeighted(base, 1, warped_image, 1, 0)

    return base


def create_image_with_dimensions(width, height, colour=(0, 0, 0)):
    """Creates a blank image with the specified dimensions"""
    colour = colour + (1,)
    return np.zeros((height, width, 3, 1), np.uint8) + colour


def get_monitor_dimensions():
    """Get the height and width of the users monitor. This should work on Windows, Mac, and Linux"""
    root = tk.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

def round_corners(image, radius):
    """Rounds the corner of the image by the specified radius"""
    # Create a mask with rounded corners
    mask = create_image_with_dimensions(image.shape[1], image.shape[0], (0, 0, 0))
    # Convert to a CV2 image
    mask = np.array(mask, dtype=np.uint8)
    # Convert to a 3 channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw a rectangle with rounded corners
    cv2.rectangle(mask, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    # Draw a circle in each corner
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (image.shape[1] - radius, radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (radius, image.shape[0] - radius), radius, (255, 255, 255), -1)
    cv2.circle(mask, (image.shape[1] - radius, image.shape[0] - radius), radius, (255, 255, 255), -1)
    # Fill in the inside with rectangles
    cv2.rectangle(mask, (radius, 0), (image.shape[1] - radius, image.shape[0]), (255, 255, 255), -1)
    cv2.rectangle(mask, (0, radius), (image.shape[1], image.shape[0] - radius), (255, 255, 255), -1)

    # Apply the mask to the image
    image = cv2.bitwise_and(image, mask)
    return image
