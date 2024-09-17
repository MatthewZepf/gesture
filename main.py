import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np
import time
from collections import deque

# Initialize dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    for rect in rects:
        # Get the facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Draw the landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Green circles for landmarks

        # Track the nose (landmark 30)
        nose = shape[30]
        
        if previous_nose is not None:
            # Compute the movement vector
            movement_vector = np.array(nose) - np.array(previous_nose)
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
            end_point = (nose[0], nose[1])
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
        previous_nose = nose

    return previous_nose

def main():
    # Start the video stream
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    # Set resolution to 1280x720 (HD) or 1920x1080 (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    previous_nose = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)

        # Process the frame
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