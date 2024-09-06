
import cv2
# import sys

from fer import FER

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the FER model
emotion_detector = FER()

# Loop to continuously get frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Detect emotions in the frame
    emotions = emotion_detector.detect_emotions(frame)
    
    # If emotions were detected
    if emotions:
        # Get the top emotion with the highest confidence
        top_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        
        # Get bounding box coordinates for the face
        (x, y, w, h) = emotions[0]['box']
        
        # Draw a rectangle around the face 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Put the emotion label above the rectangle
        cv2.putText(frame, top_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame with detected emotions
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
