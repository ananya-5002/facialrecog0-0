import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mock function to simulate anti-spoofing detection
def is_real_face():
    # Simulate an accuracy percentage for demonstration purposes
    accuracy = np.random.uniform(0, 100)  # Random accuracy between 0 and 100
    return accuracy

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

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
