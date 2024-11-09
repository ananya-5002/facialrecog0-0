import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to upload an image and detect faces
def upload_and_detect_faces():
    # Initialize Tkinter window and hide it
    Tk().withdraw()

    # Open a file dialog to choose an image file
    image_path = askopenfilename(title='Select an Image for Face Detection',
                                 filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])

    # Check if a file was selected
    if not image_path:
        print("No file selected.")
        return

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the selected image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load the image.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output image with detected faces
    cv2.imshow('Detected Faces', image)

    # Save the output image (optional)
    output_path = image_path.replace('.', '_detected.')
    cv2.imwrite(output_path, image)
    print(f"Output image saved as {output_path}")

    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Call the function
upload_and_detect_faces()
