"""
The main driver for the program - This is what is run when the program is started
It runs the main loop, handles the camera feed, and stores all 3D data
"""

# import body
import cv2
from modules import hands
from modules import manipulation
import numpy as np

from modules import screenspace
from modules import body

import os
import threading

import pygame

pygame.init()


colours = {
    "red": ("#F27878", "#D96B6B"),
    "yellow": ("#F2D478", "#EDC575"),
    "green": ("#65CC76", "#60B258"),
    "blue": ("#71AFE5", "#6576CC"),
    "magenta": ("#8D58B2", "#A358B2"),
    "black": ("#424242", "#242424")
}
sizes = {
    "small": 3,
    "medium": 5,
    "large": 10
}


class Driver:
    """
    The main driver for the program - This is what is run when the program is started
    """
    def __init__(
        self,
        debug=False,
        modules=[],
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        width=300,
        height=150
    ):
        self.modules = modules
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

        self.camera_frame = None
        self.current_frame = None
        self.debug = debug
        self.warp_matrix = None
        self.inverse_matrix = None

        self.videospace_stylus_coords = []
        self.stylus_coords = (0, 0)
        self.stylus_draw = None

        self.hand_video_coords = None
        self.hand_normalised_coords = None
        self.screenspace_hand_points = None
        self.full_hand_results = None
        self.full_hand_landmarks = None

        self.full_body_results = None
        self.videospace_body_coordinates = None
        self.screenspace_body_points = None

        self.screenspace_corners = None
        self.screenspace_midpoints = None
        self.screenspace_center = None

        self.previous_full_codes = [x for x in screenspace.default_full_codes]

        self.visibility = "Calibration"  # Calibration, Correcting, Accurate
        self.visibility_time = 1000

        self.mode = "normal"
        self.colour = "red"
        self.pen_size = 5
        self.monitor_dimensions = (1920, 1080)

        self.rendered_frame = None
        self.first_camera_frame = True
        self.clicked_before = None
        self.clicked = None

        self.output_size = (500, 0)
        self.frame_number = 0

    def use_monitor_display(self):
        """Uses the user's screen as the output, rather than the physical codes"""
        self.mode = "monitor"
        self.monitor_dimensions = manipulation.get_monitor_dimensions()
        self.show_corner_codes()

    @staticmethod
    def hex_to_bgr(hex_code):
        """Converts a hex code to a BGR tuple"""
        hex_code = hex_code.lstrip('#')
        length = len(hex_code)
        return tuple(reversed([int(hex_code[i:i + length // 3], 16) for i in range(0, length, length // 3)]))

    def calculate(self, width, height):
        """
            Runs all calculations for the current frame - This finds the position of the codes on the screen,
            and the position of the stylus
        """
        frame = screenspace.get_current_frame()
        self.camera_frame = frame.copy()
        if self.first_camera_frame:
            # Calculate the video ratio
            video_ratio = frame.shape[1] / frame.shape[0]
            self.output_size = (width, int(width / video_ratio))
            # Set the dimensions of the UI window
            self.first_camera_frame = False
            self.screen = pygame.display.set_mode((self.output_size[0] + 40 + 32 + 20, self.output_size[1] + 140))
            pygame.display.set_caption("Screenspace")
        dimensions = manipulation.create_image_with_dimensions(width, height)
        # Get the corners of the screen
        self.screenspace_corners, output_frame, self.previous_full_codes, self.videospace_stylus_coords, \
            self.stylus_draw, visibility = screenspace.get_screenspace_points(
                frame, frame, self.debug, self.previous_full_codes
            )
        # If the visibility has changed, reset the visibility time
        if visibility != self.visibility:
            self.visibility = visibility
            self.visibility_time = 0
        else:
            self.visibility_time += 1
        # Get the midpoints of the screen
        self.screenspace_midpoints, output_frame = screenspace.get_midpoints(
            self.screenspace_corners, frame, self.debug
        )
        self.warp_matrix = manipulation.generate_warp_matrix(dimensions, self.screenspace_corners)
        # Calculate the inverse matrix if possible
        try:
            self.inverse_matrix = np.linalg.inv(self.warp_matrix)
        except np.linalg.LinAlgError:
            self.inverse_matrix = None

        # Find the center of the screen
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        center_warp_matrix = manipulation.generate_warp_matrix(frame, self.screenspace_corners)
        self.screenspace_center = manipulation.find_new_coordinate((center_x, center_y), center_warp_matrix)
        # Find the center of the screen in the videospace
        if len(self.videospace_stylus_coords):
            new_coords = [
                manipulation.find_new_coordinate(c, self.inverse_matrix) for c in self.videospace_stylus_coords
            ]
            # Find the midpoint
            self.stylus_coords = (
                round((new_coords[0][0] + new_coords[1][0]) / 2),
                round((new_coords[0][1] + new_coords[1][1]) / 2)
            )
        else:
            self.stylus_coords = None

        output_frame = screenspace.add_screenspace_overlay(output_frame, self.screenspace_corners, self.debug)
        # If the user wants to calculate hands points
        if "hands" in self.modules:
            hand_points, self.full_hand_results = hands.get_hand_points(frame)
            self.full_hand_landmarks = hand_points
            output_frame = hands.render_hand_points(output_frame, self.full_hand_results, self.debug)
            self.screenspace_hand_points = []
            if hand_points:
                # self.raisedFingers = [hands.get_extended_fingers(hp) for hp in handPoints]
                self.hand_video_coords = [
                    hands.to_videospace_coords(h.landmark, output_frame.shape[1], output_frame.shape[0])
                    for h in hand_points
                ]
                # To work out positions on screen, multiply by the warp matrix
                self.hand_normalised_coords = []
                for hand in self.hand_video_coords:
                    self.hand_normalised_coords.append([])
                    self.screenspace_hand_points.append([])
                    for point in hand:
                        self.hand_normalised_coords[-1].append(manipulation.find_new_coordinate(
                            point,
                            self.warp_matrix
                        ))
                        if self.inverse_matrix is not None:
                            self.screenspace_hand_points[-1].append(manipulation.find_new_coordinate(
                                point,
                                self.inverse_matrix
                            ))
            output_frame = hands.render_hand_points(output_frame, self.full_hand_results, self.debug)
        # If the user wants to calculate body points
        if "body" in self.modules:
            self.full_body_results = body.get_body_points(frame)

        self.current_frame = output_frame

    @staticmethod
    def show_corner_codes():
        """Shows codes in the corner of the users screen"""
        # In each corner of the users monitor, show codes 0 through 3
        monitor_width, monitor_height = manipulation.get_monitor_dimensions()
        # Code 0 should be in the top left, and the others should follow clockwise
        positions = [
            (0, 0),
            (monitor_width - 175, 0),
            (monitor_width - 175, monitor_height - 225),
            (0, monitor_height - 225)
        ]
        for i in range(4):
            # Load the image
            image = cv2.imread(f"assets/codes/{i}.png")
            # Add a 20px white border around the edge
            image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            # Resize the image to 150x150
            image = cv2.resize(image, (150, 150))
            # Show the window in the correct position
            cv2.imshow(f"Code {i}", image)
            # Move the window to the correct position
            cv2.moveWindow(f"Code {i}", positions[i][0], positions[i][1])

    def render(self, frame, overlay=None) -> None:
        """
        Render is a slow function - so instead create a thread for it and allow processing of the next frame
        This code may not always be used, but it's here if needed
        """
        self.frame_number += 1
        self._render(frame, overlay)
        # threading.Thread(target=self._render, args=(frame, overlay)).start()

    def _render(self, frame, overlay=None) -> None:
        """Outputs the frame in the desired format"""
        cv2.imshow("Video Feed", self.camera_frame)
        if (self.mode == "normal") or True:
            output_frame = self.current_frame.copy()
            if self.visibility_time < 1_000:
                output_frame = manipulation.overlay_image(output_frame, frame, self.warp_matrix)
            # Resize to 1000 width, keeping aspect ratio
            output_frame = cv2.resize(output_frame, (1000, round(1000 * output_frame.shape[0] / output_frame.shape[1])))
            # Flip the frame horizontally
            if self.flip_horizontal:
                output_frame = cv2.flip(output_frame, 1)
            # Flip the frame vertically
            if self.flip_vertical:
                output_frame = cv2.flip(output_frame, 0)
            if overlay is not None:
                # Make overlay the same size as the output frame
                overlay = cv2.resize(overlay, (output_frame.shape[1], output_frame.shape[0]))
                # Create a mask of the overlay. Black pixels should be ignored
                mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                # Make all coloured areas of the mask completely black on the main frame
                output_frame = cv2.bitwise_and(output_frame, output_frame, mask=cv2.bitwise_not(mask))
                # Make transparent areas of the overlay completely black
                overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
                # For each channel, add the overlay to the main frame (where the mask is not black)
                for i in range(3):
                    output_frame[:, :, i] = output_frame[:, :, i] + overlay[:, :, i]
            self.rendered_frame = output_frame
            cv2.waitKey(1)
        elif self.mode == "monitor":
            ...

        # Resize the frame to fit the desired resolution
        self.rendered_frame = cv2.resize(self.rendered_frame, self.output_size, interpolation=cv2.INTER_AREA)
        # Round the corners of the rendered frame
        self.rendered_frame = manipulation.round_corners(self.rendered_frame, 25)
        image = pygame.image.frombuffer(self.rendered_frame.tostring(), self.rendered_frame.shape[:2][::-1], "BGR")
        # Add the image to the screen
        self.screen.blit(image, (20, 20))

        clicked = self.add_ui_elements()
        if clicked != self.clicked_before:
            self.clicked = clicked
            self.handle_event(clicked)
        self.clicked_before = clicked

        # Add buttons and dropdowns to the screen
        pygame.display.update()

    def handle_event(self, action):
        if action in colours:
            self.colour = action
        elif action in list(sizes.values()):
            self.pen_size = action

    def add_ui_elements(self):
        add_button((20, self.output_size[1] + 40), 100, 25, "Windowed", "#424242", "#D9D9D9", "#C4C4C4", self.screen)
        i = 0
        clicked = None
        # Spacing between each icon
        spacing = ((self.output_size[1]) // (len(colours) + len(sizes) + 3)) + 1
        # Colours
        for key, value in colours.items():
            if add_icon_button(
                (self.output_size[0] + spacing, 20 + spacing * i),
                32, 32,
                value[0], value[1],
                ("assets/Pencil.png" if key == self.colour else "assets/Blank.png"),
                self.screen
            ):
                clicked = key
            i += 1
        # Sizes
        for key, value in sizes.items():
            if add_icon_button(
                (self.output_size[0] + spacing, 20 + spacing * i),
                32, 32,
                "#000000", "#000000",
                "assets/" + key.capitalize() + ("Grey" if value == self.pen_size else "") + ".png",
                self.screen
            ):
                clicked = value
            i += 1
        for icon in ["Undo", "Redo", "Help"]:
            if add_icon_button(
                (self.output_size[0] + spacing, 20 + spacing * i),
                32, 32,
                "#000000", "#000000",
                "assets/" + icon + ".png",
                self.screen
            ):
                clicked = icon
            i += 1
        return clicked

    @staticmethod
    def kill():
        """Ends the program gracefully"""
        cv2.destroyAllWindows()
        screenspace.kill()
        pygame.quit()


def hex_to_rgb(hex_code):
    """Converts a hex code to an RGB tuple"""
    # Remove the # from the start of the hex code
    hex_code = hex_code[1:]
    # Split the hex code into 3 parts
    hex_code = [hex_code[:2], hex_code[2:4], hex_code[4:]]
    # Convert each part to an integer
    hex_code = [int(n, 16) for n in hex_code]
    return tuple(hex_code)


def add_button(position, width, height, text, colour, background_colour, background_colour_hover, screen):
    """Adds a button to the screen"""
    colour, background_colour, background_colour_hover = hex_to_rgb(colour), hex_to_rgb(background_colour), hex_to_rgb(background_colour_hover)

    # Create the background
    button_background = pygame.Surface((width, height))
    # Create the button rect
    button_rect = pygame.Rect(position[0], position[1], width, height)

    # If the mouse is over the button
    if button_rect.collidepoint(pygame.mouse.get_pos()):
        # Change the background colour
        button_background.fill(background_colour_hover)
    else:
        # Change the background colour
        button_background.fill(background_colour)

    # Create the text
    button_text = pygame.font.Font("assets/Roboto-regular.ttf", 16).render(text, True, colour)
    # Add the text to the background
    button_background.blit(button_text, (width // 2 - button_text.get_width() // 2, height // 2 - button_text.get_height() // 2))
    screen.blit(button_background, position)
    return button_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]

def add_icon_button(position, height, width, background_colour, background_colour_hover, image_path, screen):
    """Creates a button in the desired location which displays an image"""
    background_colour, background_colour_hover = hex_to_rgb(background_colour), hex_to_rgb(background_colour_hover)

    # Create the background
    button_background = pygame.Surface((width, height))
    # Create the button rect
    button_rect = pygame.Rect(position[0], position[1], width, height)

    # If the mouse is over the button
    if button_rect.collidepoint(pygame.mouse.get_pos()):
        # Change the background colour
        button_background.fill(background_colour_hover)
    else:
        # Change the background colour
        button_background.fill(background_colour)

    # Load the image
    image = pygame.image.load(image_path)
    # Resize the image to fit the button
    image = pygame.transform.scale(image, (width, height))
    # Add the image to the background
    button_background.blit(image, (0, 0))
    screen.blit(button_background, position)
    return button_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]
