import sys
import cv2
import numpy as np
from modules.driver import Driver


# Find command arguments
args = sys.argv[1:]
flags = {
    "-m": "--monitor", "-H": "--horizontal", "-V": "--vertical", "-d": "--debug", "-h": "--help"
}
# Create a list of flags that are set, in their long form
flags = [flags[arg] if arg in flags else arg for arg in args]


width = 200  # Camera width
height = 100  # Camera height
scale = 5  # Scale the output by this amount
width, height = width * scale, height * scale  # Adjust the width and height to the scale

driver = Driver(debug=("--debug" in flags), modules=["hands"],
                flip_horizontal=("--horizontal" in flags), flip_vertical=("--vertical" in flags), height=height, width=width, use_pygame=False)

# For the background image, load assets/TestImage.png into cv2
background = cv2.imread("assets/TestImage.png")
# Replace all pixels with an r, g, or b value less than 2 with 2
background[background < 2] = 2

exit_flag = False

while not exit_flag:
    # Calculate new matrices
    driver.calculate(background.shape[1], background.shape[0])

    driver.render(background)
