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


# Connections
connections = [ (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (5, 1), (5, 17) ]
# Fingers
fingers = [
    [(1, 2), (2, 3), (3, 4)],
    [(5, 6), (6, 7), (7, 8)],
    [(9, 10), (10, 11), (11, 12)],
    [(13, 14), (14, 15), (15, 16)],
    [(17, 18), (18, 19), (19, 20)]
]


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
            # List the 4 points that make up each finger into a 2D array. The thumb is 1 to 4, the index finger is 5 to 8, etc.
            fingersUp = []
            for n in range(5):
                # Create a list of points from index 4n+1 to 5n+4
                finger = [handLms.landmark[4*n+1], handLms.landmark[4*n+2], handLms.landmark[4*n+3], handLms.landmark[4*n+4]]
                # Check if the finger is up. This is done by checking the dot product of the vector from points 0 to 1, and 2 to 3.
                vec1, vec2 = (finger[1].x - finger[0].x, finger[1].y - finger[0].y), (finger[3].x - finger[2].x, finger[3].y - finger[2].y)
                dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                fingersUp.append(dot > 0)
            # Non finger connections
            for connection in connections:
                cv2.line(frame, (int(handLms.landmark[connection[0]].x * frame.shape[1]), int(handLms.landmark[connection[0]].y * frame.shape[0])), (int(handLms.landmark[connection[1]].x * frame.shape[1]), int(handLms.landmark[connection[1]].y * frame.shape[0])), (255, 255, 255), 2)
            # Fingers
            for finger in fingers:
                for connection in finger:
                    cv2.line(frame, (int(handLms.landmark[connection[0]].x * frame.shape[1]), int(handLms.landmark[connection[0]].y * frame.shape[0])), (int(handLms.landmark[connection[1]].x * frame.shape[1]), int(handLms.landmark[connection[1]].y * frame.shape[0])), (0, 255, 0) if fingersUp[fingers.index(finger)] else (255, 0, 0), 2)
            # Finger points (all points > 0 and not 1 above a multiple of 4)
            for n in range(21):
                if n % 4 != 1 and n > 0:
                    cv2.circle(frame, (int(handLms.landmark[n].x * frame.shape[1]), int(handLms.landmark[n].y * frame.shape[0])), 5, (0, 255, 0) if fingersUp[(n-2)//4] else (255, 0, 0), cv2.FILLED)
            # Thumb point
            cv2.circle(frame, (int(handLms.landmark[1].x * frame.shape[1]), int(handLms.landmark[1].y * frame.shape[0])), 5, (255, 255, 255), cv2.FILLED)
            # Wrist point
            cv2.circle(frame, (int(handLms.landmark[0].x * frame.shape[1]), int(handLms.landmark[0].y * frame.shape[0])), 5, (255, 255, 255), cv2.FILLED)
    # Convert the image back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Show the frame
    cv2.imshow("Frame", frame)
    # If the user pressed "q", quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
