import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image (replace 'path_to_your_image.jpg' with your image path)
image_path = 'C:/Users/SHREYA ANANYA/Downloads/GA311881_Bill_Gates.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()  # Close all OpenCV windows
