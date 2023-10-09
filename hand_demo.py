import mediapipe as mp
from imutils.video import VideoStream
import cv2


# Start the video stream
vs = VideoStream(src=0).start()

# Start the mediapipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=6,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.2
)
mpDraw = mp.solutions.drawing_utils


# Loop over the frames from the video stream
while True:
    # Get the frame from the video stream and convert it to RGB
    frame = vs.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect the hands
    results = hands.process(frame)
    # Convert to a 2D array of 3D points (list of hands, containing a list of (x, y, z) points)
    landmarks = results.multi_hand_landmarks
    # If there are any hands detected, draw them
    if landmarks:
        # Loop over the hands
        for handLms in landmarks:
            # Draw the landmarks
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    # Convert the image back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Show the frame
    cv2.imshow("Frame", frame)
    # If the user pressed "q", quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
