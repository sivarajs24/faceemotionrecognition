import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model_file_30epochs.h5')

# Load the pre-trained face detection model
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary to label all emotions
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Read the image file
image_path = 'data01.jpg'
frame = cv2.imread(image_path)

# Check if the image was loaded successfully
if frame is None:
    print(f"Error: Could not read image file {image_path}. Check the file path and try again.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

for x, y, w, h in faces:
    # Extract the region of interest (the face)
    sub_face_img = gray[y:y + h, x:x + w]
    resized = cv2.resize(sub_face_img, (48, 48))
    normalize = resized / 255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    
    # Predict the emotion
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    print(label)
    
    # Draw rectangles around the face and label it with the predicted emotion
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the image with the detected faces and labels
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
