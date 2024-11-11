import cv2

# Set the absolute path to the haarcascade file
face_cascade_path = 'D:\\Personal\\ml-course\\Mini_project\\haarcascade_frontalface_default.xml'

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Initialize label as "Real" by default
label = "Real"

# Start the webcam
cap = cv2.VideoCapture(0)

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

    # Draw rectangles around detected faces and display the label
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the label ("Real" or "Fake")
        cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the frame with detection and label
    cv2.imshow('Webcam Face Detection', frame)

    # Check if 'r' key is pressed to toggle the label
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # Toggle label between "Real" and "Fake"
        label = "Fake" if label == "Real" else "Real"
    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
