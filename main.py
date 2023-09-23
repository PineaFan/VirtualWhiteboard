#!/usr/bin/env python
"""
\033[34mScreenSpace Virtual Whiteboard (By GitHub/@PineaFan)
A program allowing the user to draw on the screen using their hands\033[0m

Flags:
    \033[32m-m, --monitor: Use the monitor as a display, rather than physical codes
    \033[32m-H, --horizontal: Flip the output horizontally
    \033[32m-V, --vertical: Flip the output vertically
    \033[33m-d, --debug: Show debug information
    \033[31m-h, --help: Show help\033[0m
"""

import sys
import cv2
import numpy as np
from modules.driver import Driver
from modules.hands import (Fist, HandModel, IndexFinger, MiddleFinger, Peace,
                           PinkyFinger, RingFinger, Spread, hand_to_name, get_extended_fingers)

from modules.login import login

uid = login()
if uid is None:
    print("Login failed")
    sys.exit()


# Find command arguments
args = sys.argv[1:]
flags = {
    "-m": "--monitor", "-H": "--horizontal", "-V": "--vertical", "-d": "--debug", "-h": "--help"
}
# Create a list of flags that are set, in their long form
flags = [flags[arg] if arg in flags else arg for arg in args]

if "--help" in flags:
    print("Usage: python3 main.py [flags]\n\n" + __doc__)
    sys.exit()


MAX_HANDS = 2

width = 200  # Camera width
height = 100  # Camera height
scale = 5  # Scale the output by this amount
width, height = width * scale, height * scale  # Adjust the width and height to the scale

driver = Driver(debug=("--debug" in flags), modules=["hands"],
                flip_horizontal=("--horizontal" in flags), flip_vertical=("--vertical" in flags), height=height, width=width)


class Colours:
    """A list of colours the program can use at any point"""
    red: tuple = driver.hex_to_bgr("#F27878")
    green: tuple = driver.hex_to_bgr("#A1CC65")
    blue: tuple = driver.hex_to_bgr("#6576CC")
    yellow: tuple = driver.hex_to_bgr("#E6DC71")
    cyan: tuple = driver.hex_to_bgr("#71AEF5")
    magenta: tuple = driver.hex_to_bgr("#A358B3")
    white: tuple = driver.hex_to_bgr("#FFFFFF")
    black: tuple = driver.hex_to_bgr("#020202")
    transparent: tuple = driver.hex_to_bgr("#000000")

# Defines which hand models refer to which tool
actions = {
    IndexFinger().name: "draw",
    Peace().name: "line",
    Spread().name: "erase",
    MiddleFinger().name: "quit"
}


# Create a background
background_colour = Colours.white
background = np.zeros((height, width, 3), np.uint8)
background[:] = background_colour

# This will be an overlay over the background
current_drawing = np.zeros((height, width, 3), np.uint8)
current_drawing[:] = background_colour

# This is what the user is currently drawing, such as a line, and can be cleared
# If the user draws a line, it will be added here as a "preview"
current_motion = np.zeros((height, width, 3), np.uint8)
current_motion[:] = Colours.transparent

# This is the current path being drawn by a user.
# It is temporarily stored here while the user is drawing, and is added to the current_drawing when they stop
current_path = np.zeros((height, width, 3), np.uint8)
current_path[:] = Colours.transparent

# What the user is currently doing, such as draw, line, erase, etc.
current_action = [None for _ in range(MAX_HANDS)]
known_hands = [None for _ in range(MAX_HANDS)]
previous_hands = [None for _ in range(MAX_HANDS)]

undo_stack = [current_drawing.copy()]
redo_stack = []


statuses = {
    "Calibration": [Colours.red, "", 20],
    "Correcting": [Colours.yellow, "", 5],
    "Accurate": [Colours.green, "", 5]
}

current_overlay = None
last_visibility = False

exit_flag = False

if "--monitor" in flags or "-m" in flags:
    driver.use_monitor_display()

current_paths = [{"pathType": None, "path": []} for _ in range(MAX_HANDS)]
last_clicked = None

render_current_path = False

while not exit_flag:
    current_frame = background.copy()
    if current_overlay is None and driver.camera_frame is not None:
        current_overlay = np.zeros((driver.camera_frame.shape[0], driver.camera_frame.shape[1], 3), np.uint8)

    # Preprocessing
    # Overlay the current drawing. Use black areas as a mask
    for i in range(3):
        current_frame[:, :, i] = np.where(current_drawing[:, :, i] == 0, current_frame[:, :, i], current_drawing[:, :, i])

    # Calculate new matrices
    driver.calculate(background.shape[1], background.shape[0])

    # Render the stylus
    if driver.stylus_coords is not None:
        cv2.circle(
            current_drawing if driver.stylus_draw else current_overlay,
            (round(driver.stylus_coords[0]), round(driver.stylus_coords[1])), 3, (255, 0, 255), -1
        )
    # If there are hands on screen
    elif driver.screenspace_hand_points:
        to_render = [4, 8, 12, 16, 20]  # Render a dot at the tip of each finger if in debug mode TODO

        # Loop over each hand
        for hand_index, hand_data in enumerate(driver.full_hand_landmarks):
            # If it's None, skip it
            if hand_data is None:
                continue
            # For each finger
            raised = [finger > 0 for finger in get_extended_fingers(hand_data)]
            hand_model = HandModel(raised)
            hand_model.name = hand_to_name(raised)
            if hand_index >= len(previous_hands) or hand_index >= len(known_hands):
                continue
            if previous_hands[hand_index] is None:
                previous_hands[hand_index] = [hand_model, hand_data, 0]
            if previous_hands[hand_index][0].name == hand_model.name:
                previous_hands[hand_index] = [hand_model, hand_data, previous_hands[hand_index][2] + 1]
            else:
                previous_hands[hand_index] = [hand_model, hand_data, 0]
            if previous_hands[hand_index][2] > 10:
                known_hands[hand_index] = previous_hands[hand_index][0]
            else:
                known_hands[hand_index] = None

    # Check if the user has released a button (last_clicked (old) vs driver.clicked (current))
    if last_clicked is not None and not driver.clicked:
        if last_clicked == "Undo":
            # Remove the last item from the undo stack
            # Add it to the redo stack
            if len(undo_stack) > 1:
                redo_stack.append(undo_stack.pop())
            current_drawing = undo_stack[-1].copy()
        elif last_clicked == "Redo":
            # Remove the last item from the redo stack
            # Add it to the undo stack
            if len(redo_stack) > 0:
                undo_stack.append(redo_stack.pop())
            current_drawing = undo_stack[-1].copy()
    last_clicked = driver.clicked
    for hand_index, hand in enumerate(known_hands):
        if hand is None:
            continue
        if len(driver.screenspace_hand_points) <= hand_index:
            continue
        match actions.get(hand_to_name(hand.value[:5]), None):
            case "quit":
                exit_flag = True
            case "draw":
                x = round(driver.screenspace_hand_points[hand_index][8][0])
                y = round(driver.screenspace_hand_points[hand_index][8][1])
                new_points = []
                # cv2.circle(current_path, (x, y), driver.pen_size, getattr(Colours, driver.colour), -1)
                # Find the distance between the last point and the current point
                if len(current_paths[hand_index]["path"]) > 0:
                    dist = np.sqrt((x - current_paths[hand_index]["path"][-1][0]) ** 2 + (y - current_paths[hand_index]["path"][-1][1]) ** 2)
                    # Add x midpoints between the last point and the current point. The more midpoints, the smoother the line
                    # The number of midpoints is the distance divided by the pen size
                    for i in range(round(dist / driver.pen_size)):
                        new_points.append((
                            round(current_paths[hand_index]["path"][-1][0] + (x - current_paths[hand_index]["path"][-1][0]) / (i + 1)),
                            round(current_paths[hand_index]["path"][-1][1] + (y - current_paths[hand_index]["path"][-1][1]) / (i + 1))
                        ))
                new_points.append((x, y))
                current_paths[hand_index]["path"].extend(new_points)
                for point in new_points:
                    cv2.circle(current_path, point, driver.pen_size, getattr(Colours, driver.colour), -1)
                current_paths[hand_index]["path"].append((x, y))
                render_current_path = True
            case "erase":
                # If the eraser has only just been activated, set the current path to the current drawing
                if current_paths[hand_index]["pathType"] is None:
                    current_paths[hand_index]["pathType"] = "eraser"
                    current_paths[hand_index]["path"] = []
                    current_path = current_drawing.copy()
                focus_about = [0, 8, 20]
                eraser_size = driver.pen_size * 4
                # Avoid list index out of range errors
                if not len(driver.screenspace_hand_points) or True:
                    for i in range(len(focus_about)):
                        if focus_about[i] >= len(driver.screenspace_hand_points[hand_index]):
                            focus_about[i] = len(driver.screenspace_hand_points[hand_index]) - 1

                    x, y = round((driver.screenspace_hand_points[hand_index][focus_about[0]][0] + driver.screenspace_hand_points[hand_index][focus_about[1]][0] + driver.screenspace_hand_points[hand_index][focus_about[2]][0]) / 3),\
                           round((driver.screenspace_hand_points[hand_index][focus_about[0]][1] + driver.screenspace_hand_points[hand_index][focus_about[1]][1] + driver.screenspace_hand_points[hand_index][focus_about[2]][1]) / 3)
                    # cv2.circle(current_path, focus, eraser_size, Colours.white, -1)
                    new_points = []
                    # Find the distance between the last point and the current point
                    if len(current_paths[hand_index]["path"]) > 0:
                        dist = np.sqrt((x - current_paths[hand_index]["path"][-1][0]) ** 2 + (y - current_paths[hand_index]["path"][-1][1]) ** 2)
                        # Add x midpoints between the last point and the current point. The more midpoints, the smoother the line
                        # The number of midpoints is the distance divided by the pen size
                        for i in range(round(dist / eraser_size)):
                            new_points.append((
                                round(current_paths[hand_index]["path"][-1][0] + (x - current_paths[hand_index]["path"][-1][0]) / (i + 1)),
                                round(current_paths[hand_index]["path"][-1][1] + (y - current_paths[hand_index]["path"][-1][1]) / (i + 1))
                            ))
                    new_points.append((x, y))
                    current_paths[hand_index]["path"].extend(new_points)
                    for point in new_points:
                        cv2.circle(current_path, point, eraser_size, Colours.white, -1)
                    # Show an outline of the eraser on the current_motion
                    # To do this, draw a filled circle, then draw a transparent circle over the top
                    cv2.circle(current_motion, (x, y), eraser_size, Colours.magenta, -1)
                    cv2.circle(current_motion, (x, y), eraser_size - 2, Colours.transparent, -1)
                    current_paths[hand_index]["path"].append((x, y))
                    render_current_path = True
    if all([hand is None for hand in known_hands]):
        if render_current_path:
            # Add the current path to the current_drawing
            current_drawing[:, :, :] = np.where(
                current_path[:, :, :] == 0, current_drawing[:, :, :], current_path[:, :, :]
            )
            # Add the current drawing to the undo stack
            undo_stack.append(current_drawing.copy())
            current_path = np.zeros((height, width, 3), np.uint8)
            current_path[:] = Colours.transparent
            render_current_path = False
            current_paths = [{"pathType": None, "path": []} for _ in range(MAX_HANDS)]

    # Create a mask from the current_motion, where all non black pixels become white
    mask = cv2.inRange(current_motion, Colours.transparent, Colours.transparent)
    # Apply the mask to the current_frame, so white pixels on the mask make the current_frame black
    current_frame = cv2.bitwise_and(current_frame, current_frame, mask=mask)

    for i in range(3):
        if render_current_path:
            current_frame[:, :, i] = np.where(
                current_path[:, :, i] == 0, current_frame[:, :, i], current_path[:, :, i]
            )
        current_frame[:, :, i] = np.where(
            current_motion[:, :, i] == 0, current_frame[:, :, i], current_motion[:, :, i]
        )

    # cv2.imshow("current_path", current_path)
    current_motion[:] = Colours.transparent

    # Only update when the status has been the same for 10 frames
    if ((driver.visibility_time > 3 and driver.visibility != last_visibility) or driver.visibility_time > 10) and current_overlay is not None:
        last_visibility = driver.visibility
        status_name = driver.visibility
        status = statuses[status_name]
        if status_name == "Accurate":
            current_overlay[:, :, :] = Colours.transparent
        cv2.rectangle(current_overlay, (0, 0), (current_overlay.shape[1], status[2]), status[0], -1)
        # Add text to the overlay, 20px high and white
        cv2.putText(current_overlay, status[1], (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colours.white, 1, cv2.LINE_AA)
        # cv2.imshow("current_overlay", current_overlay)

    driver.render(current_frame, current_overlay)
