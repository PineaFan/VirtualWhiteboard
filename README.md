# Screenspace
*By Pinea*


Screenspace allows you to add images into the camera, between preset markers

[Generate codes](https://chev.me/arucogen/) - Codes should be 5x5. From top left clockwise, codes 0, 1, 2 and 3 should be used

### Examples

```py
# Show a transparent rectangle around the end of the pointer finger
from driver import Driver
import cv2
import time
import manipulation
import numpy as np

driver = Driver(debug=True, modules=["hands"])  # Make sure to add the hands module
background = np.zeros([1, 1, 1], dtype=np.uint8)  # Create a black background
background.fill(255)  # Fill the background with white
background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
background = cv2.resize(background, (150, 100))  # Resize the background to 150x100

while True:
    frame = background.copy()  # Create a copy of the background, so it can be modified without affecting the original
    driver.calculate(background.shape[1], background.shape[0])  # Run all calculations - hand coordinates, matrices etc.
    # Show point 8 from the screenspace hand points (index finger tip)
    if driver.screenspaceHandPoints:
        point = driver.screenspaceHandPoints[8]
        cv2.rectangle(frame, (5, round(point[1] - 10)), (10, round(point[1] + 10)), (1, 1, 1, 1), -1)

    driver.render(frame)  # Frame will be added to the webcam stream, with its corners stretched to the tags (if in the correct place)
```
