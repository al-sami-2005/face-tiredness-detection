import cv2
from deepface import DeepFace

# Load the built-in face and eye detectors from OpenCV
# (These come pre-installed with opencv-python, so no extra downloads needed!)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

# Variables for logic
frame_counter = 0
eyes_closed_frames = 0
TIRED_THRESHOLD = 15  # If eyes are not seen for 15 frames, assume closed/tired
current_emotion = "..."
current_age = "..."
is_tired = False

print("Starting... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no face is detected, we can't detect eyes
    if len(faces) == 0:
        eyes_closed_frames = 0
    
    for (x, y, w, h) in faces:
        # Draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Focus on the face area to find eyes (Optimization)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # 2. Detect Eyes within the face
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        
        # LOGIC: If face is present but NO eyes are detected, eyes might be closed.
        if len(eyes) == 0:
            eyes_closed_frames += 1 
        else:
            eyes_closed_frames = 0 # Eyes are open
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2) 
            
        # Check Tiredness
        if eyes_closed_frames > TIRED_THRESHOLD:
            is_tired = True
        else:
            is_tired = False

        # 3. DeepFace Analysis (Run every 10 frames to stop lag)
        if frame_counter % 10 == 0:
            try:
                # We interpret the face region only
                face_img = frame[y:y+h, x:x+w]
                results = DeepFace.analyze(face_img, actions=['emotion', 'age'], enforce_detection=False)
                current_emotion = results[0]['dominant_emotion']
                current_age = results[0]['age']
            except:
                pass

        # Display Info on screen
        cv2.putText(frame, f"Emotion: {current_emotion}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        cv2.putText(frame, f"Age: {current_age}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    # 4. Global Warning System
    if is_tired:
        cv2.putText(frame, "WARNING: TIRED / EYES CLOSED", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    frame_counter += 1
    cv2.imshow('Tiredness & Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()