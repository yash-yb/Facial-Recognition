import cv2

def land_marks():
    '''
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture from the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Landmarks Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()'''

import cv2
import numpy as np
import mediapipe as mp

# Step 1: Initialize modern MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Step 2: Define specific connectivity maps for Face Mesh Contours manually
# (Since mp.solutions metadata is missing from your environment)
LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), 
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 74), (74, 42), 
    (42, 88), (88, 95), (95, 144), (144, 178), (178, 87), (87, 14), (14, 317), 
    (317, 402), (402, 373), (373, 324), (324, 304), (304, 303), (303, 295), (295, 289), 
    (289, 409), (409, 291), (78, 95), (88, 178), (87, 317), (402, 324)
]
LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), 
    (382, 362), (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), 
    (384, 398), (398, 362)
]
RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), 
    (155, 133), (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), 
    (157, 173), (173, 133)
]
FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), 
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), 
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), 
    (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), 
    (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]
ALL_CONNECTIONS = LIPS + LEFT_EYE + RIGHT_EYE + FACE_OVAL

def draw_landmarks_opencv(image, landmarks, connections, color=(0, 255, 0), thickness=1):
    """Draws custom facial map lines onto image matrix using coordinates scaled to resolution."""
    h, w, _ = image.shape
    
    # Draw connection lines
    for connection in connections:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
            pt1 = landmarks[pt1_idx]
            pt2 = landmarks[pt2_idx]
            
            # Map normalized coordinates (0.0 to 1.0) back to raw integer pixel spaces
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Step 3: Configure FaceLandmarker Options referencing downloaded file
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Start webcam capture configuration
cap = cv2.VideoCapture(0)

# Step 4: Open context wrapper safely
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame for native mirror preview matching old code
        frame = cv2.flip(frame, 1)

        # Convert layout from default BGR to standard RGB processing channel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Format numpy frame matrices into strict MediaPipe native Image wrappers
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process inference tasks
        detection_result = landmarker.detect(mp_image)

        # Draw structural overlay mappings if facial structures appear in viewport
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                draw_landmarks_opencv(
                    image=frame, 
                    landmarks=face_landmarks, 
                    connections=ALL_CONNECTIONS, 
                    color=(220, 220, 220), # Off-white mapping structural style color
                    thickness=1
                )

        # Render display layout framework frame
        cv2.imshow('Facial Landmarks Tracker (MediaPipe 1.0.0 Compatible)', frame)

        # Escapes layout execution using the 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
