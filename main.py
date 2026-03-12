import cv2
from deepface import DeepFace

# Load the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting video... Press 'q' to quit.")

while True:
    # 1. Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # 2. Analyze the frame with DeepFace
    # Note: DeepFace is heavy. Running it on every single frame will be slow (laggy).
    # For a smoother video, we use a try-except block to handle frames where no face is found.
    try:
        # actions=['emotion', 'age'] tells it specifically what to look for
        # enforce_detection=False prevents it from crashing if no face is seen
        results = DeepFace.analyze(frame, actions=['emotion', 'age'], enforce_detection=False)
        
        # DeepFace returns a list of dictionaries (one for each face detected)
        for face in results:
            # Get the coordinates of the face
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            
            # Get the data
            emotion = face['dominant_emotion']
            age = face['age']
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare the text to display
            text = f"{emotion}, {age} yrs"
            
            # Put the text above the rectangle
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)
            
    except Exception as e:
        print(f"Error: {e}")
        pass

    # 3. Display the resulting frame
    cv2.imshow('Face Emotion & Age Detection', frame)

    # 4. Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()