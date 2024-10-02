import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import time
from collections import deque
import datetime
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
import pyautogui


# Initialize dlib face detector and facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")

# Initialize parameters
time_interval = 0.30  # Time window for average vector calculation in seconds
magnitude_threshold = 0.01  # Threshold magnitude to trigger action
vector_window = deque()  # Store vectors with timestamps
last_trigger_time = 0
action_cooldown = 1.0  # Cooldown period of 1 second


def vector_magnitude(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2)

def average_vector(vectors):
    if not vectors:
        return np.array([0, 0])
    return np.mean(vectors, axis=0)

def clean_old_vectors(current_time):
    while vector_window and (current_time - vector_window[0][0]) > time_interval:
        vector_window.popleft()

def determine_direction(vector):
    if(abs(vector[0]) < abs(vector[1])):
        if(vector[1] < 0):
            return "Up"
        elif(vector[1] > 0):
            return "Down"
    elif vector[0] > 0:
        return "Left"
    elif vector[0] < 0:
        return "Right"
    
    return "Center"

def detect_and_draw_landmarks(frame, mesh_results, predictor):
    # return landmarks if detected, otherwise return None, also draw the landmarks on the frame
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return mesh_results.multi_face_landmarks

def calculate_movement(landmarks, previous_landmarks, current_time, frame):
    # Calculate the movement vector based on the landmarks
    # Extract nose coordinates from landmarks
    # print(type(landmarks))
    # print(landmarks.landmark[1])
    # print(landmarks)
    global last_trigger_time
    nose_x, nose_y = landmarks.landmark[1].x, landmarks.landmark[1].y
    previous_nose_x, previous_nose_y = previous_landmarks.landmark[1].x, previous_landmarks.landmark[1].y

    # Calculate movement vector from previous nose coordinates
    movement_vector = (nose_x - previous_nose_x, nose_y - previous_nose_y)

    # Calculate vector magnitude
    vector_magnitude_value = vector_magnitude(movement_vector)

    # Add current vector and timestamp to the deque
    vector_window.append((current_time, movement_vector))

    # Clean old vectors
    clean_old_vectors(current_time)

    # Compute average vector over the time window
    avg_vector = average_vector([vec for _, vec in vector_window])
    avg_magnitude = vector_magnitude(avg_vector)

    # Draw movement vector
    # Get the frame dimensions
    h, w, _ = frame.shape

    # Convert normalized coordinates to actual pixel coordinates
    start_point = (int(previous_nose_x * w), int(previous_nose_y * h))
    end_point = (int(nose_x * w), int(nose_y * h))
    cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Red arrow for movement

    # Determine direction based on average vector
    direction = determine_direction(avg_vector)

    if avg_magnitude > magnitude_threshold and (current_time - last_trigger_time) > action_cooldown:
        print(avg_vector[0])
        print(avg_vector[1])
        if direction == "Left":
            print("Trigger Copy (Ctrl/Command + C)")
            pyautogui.hotkey('command', 'c')
        elif direction == 'Right':
            print("Trigger Paste (Command + V)")
            pyautogui.hotkey('command', 'v')
        elif direction == 'Up':
            print("Trigger Highlight All (Command + A)")
            pyautogui.hotkey('command', 'a')
        elif direction == "Down":
            print("Trigger Undo(Command + z)")
            pyautogui.hotkey('command', 'z')
        last_trigger_time = current_time


def process_frame(frame, previous_landmarks):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using FaceDetection
    face_detection_results = face_detection.process(rgb_frame)

    if face_detection_results.detections:
        for detection in face_detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Draw bounding box

            # Get facial landmarks using FaceMesh
            landmarks = detect_and_draw_landmarks(frame, face_mesh.process(rgb_frame), None)
            landmarks = landmarks[0] if landmarks else None

            # Calculate movement vector based on the landmarks
            if previous_landmarks and landmarks:
                calculate_movement(landmarks, previous_landmarks, time.time(), frame)

    # if landmarks not none, return landmarks, otherwise return previous_landmarks
    return landmarks if landmarks else previous_landmarks

def main():
    cap = cv2.VideoCapture(0)
    previous_landmarks = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Process the frame
        if ret:
            frame = imutils.resize(frame, width=720)
            previous_landmarks = process_frame(frame, previous_landmarks)

            # Display the resulting frame
            cv2.imshow("Head Movement Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
