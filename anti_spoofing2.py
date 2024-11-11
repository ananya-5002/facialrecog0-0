import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

switch_state = True  

def change_state():
    global switch_state
    switch_state = not switch_state

# function to simulate anti-spoofing detection
def is_real_face():
    if switch_state:
        return np.random.uniform(50, 100)  # Real with higher accuracy
    else:
        return np.random.uniform(0, 50)  # Fake with lower accuracy

# Start the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces and determine if they are real or fake
    for (x, y, w, h) in faces:
        # Get the accuracy percentage for the detected face
        accuracy = is_real_face()
        label = "Real" if accuracy > 50 else "Fake"  # Assuming 50% as a threshold

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Prepare the label with accuracy
        label_text = f"{label} ({accuracy:.2f}%)"
        # Add the label to the image
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Face Detection with Anti-Spoofing', frame)

   
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'): 
        change_state()

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
