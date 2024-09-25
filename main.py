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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize dlib face detector and facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks_GTX.dat")

# Initialize parameters
time_interval = 0.30  # Time window for average vector calculation in seconds
magnitude_threshold = 20.0  # Threshold magnitude to trigger action
vector_window = deque()  # Store vectors with timestamps

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
    if vector[0] > 0:
        return "Right"
    elif vector[0] < 0:
        return "Left"
    return "Center"

def process_frame(frame, previous_nose):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_detection_results = face_detection.process(rgb_frame)

    if face_detection_results.detections:
        for detection in face_detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Draw bounding box

            # Get facial landmarks using FaceMesh
            mesh_results = face_mesh.process(rgb_frame)
            if mesh_results.multi_face_landmarks:
                # Access the nose landmark (index 1 for the tip of the nose)
                nose = mesh_results.multi_face_landmarks[0].landmark[1]

                # Scale the landmark position to the frame size
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                if previous_nose is not None:
                    # Compute the movement vector
                    movement_vector = np.array([nose_x, nose_y]) - np.array(previous_nose)
                    current_time = time.time()
                    vector_magnitude_value = vector_magnitude(movement_vector)

                    # Add current vector and timestamp to the deque
                    vector_window.append((current_time, movement_vector))

                    # Clean old vectors
                    clean_old_vectors(current_time)

                    # Compute average vector over the time window
                    avg_vector = average_vector([vec for _, vec in vector_window])
                    avg_magnitude = vector_magnitude(avg_vector)

                    # Draw movement vector
                    start_point = (previous_nose[0], previous_nose[1])
                    end_point = (nose_x, nose_y)
                    cv2.arrowedLine(frame, start_point, end_point, (255, 0, 0), 2)  # Red arrow for movement

                    # Determine direction based on average vector
                    direction = determine_direction(avg_vector)

                    # Print direction if the average magnitude exceeds the threshold
                    if avg_magnitude > magnitude_threshold:
                        print(f"Detected significant movement: {direction}, Average Vector: {avg_vector}")

                    # Debugging: Print vectors and times to a file
                    with open('movement_log.txt', 'a') as log_file:
                        log_file.write(f"{current_time}, {movement_vector}, {vector_magnitude_value}, {avg_vector}, {avg_magnitude}, {direction}\n")

                # Update previous nose position
                previous_nose = [nose_x, nose_y]

    return previous_nose
def main():
    cap = cv2.VideoCapture(0)
    previous_nose = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Process the frame
        if ret:
            frame = imutils.resize(frame, width=720)
            previous_nose = process_frame(frame, previous_nose)

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
