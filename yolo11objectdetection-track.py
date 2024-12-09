import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from tensorflow.keras.models import load_model


# Load gender classification model
gender_model = load_model("E:\yolo11peoplecounter-main\myvenv\models\gender_classification_model.h5")  # Replace with your model path
gender_labels = ['Male', 'Female']

# Load age classification model
age_model = load_model("E:\yolo11peoplecounter-main\myvenv\models\age_classification_model.h5")  # Replace with your model path
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71+']  # Replace based on your model

# Mouse event for debugging (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLO model
model = YOLO("yolo11s.pt")
names = model.model.names

# Open the video file
cap = cv2.VideoCapture('Maradana.mp4')

# Initialize variables
count = 0
cy1 = 370  # Enter line (green)
cy2 = 450  # Exit line (blue)
offset = 20  # Line buffer zone
track_data = {}  # Store position history, counted status, gender, and age for each track ID
enter_data = {'Male': {}, 'Female': {}}  # Dictionary to track entry counts by age group
exit_data = {'Male': {}, 'Female': {}}

# Initialize age group dictionaries
for gender in ['Male', 'Female']:
    for age in age_labels:
        enter_data[gender][age] = 0
        exit_data[gender][age] = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1  # Frame counter

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=0, conf=0.25)

    # Check if there are detections
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get bounding boxes, class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Initialize or update track data
            if track_id not in track_data:
                # Crop the detected person for gender and age classification
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (64, 64))  # Resize to model input size
                crop = np.expand_dims(crop / 255.0, axis=0)  # Normalize and expand dimensions

                # Predict gender
                gender_pred = gender_model.predict(crop)[0]
                gender = gender_labels[np.argmax(gender_pred)]

                # Predict age
                age_pred = age_model.predict(crop)[0]
                age = age_labels[np.argmax(age_pred)]

                # Initialize track data
                track_data[track_id] = {'positions': [], 'counted': False, 'gender': gender, 'age': age}

            # Update position history
            track_data[track_id]['positions'].append(cy)

            # Check movement direction
            positions = track_data[track_id]['positions']
            gender = track_data[track_id]['gender']
            age = track_data[track_id]['age']
            if len(positions) > 1 and not track_data[track_id]['counted']:
                # Movement downward (entering)
                if positions[-2] < cy1 <= positions[-1] < cy2:
                    enter_data[gender][age] += 1
                    track_data[track_id]['counted'] = True
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Movement upward (exiting)
                elif positions[-2] > cy2 >= positions[-1] > cy1:
                    exit_data[gender][age] += 1
                    track_data[track_id]['counted'] = True
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw bounding boxes and information
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c} {gender}, {age}', (x1, y1), 1, 1)

    # Draw enter and exit lines
    cv2.line(frame, (270, cy1), (900, cy1), (0, 255, 0), 2)  # Enter line (green)
    cv2.line(frame, (270, cy2), (900, cy2), (255, 255, 0), 2)  # Exit line (blue)

    # Display counts on the frame
    y_offset = 60
    for gender in ['Male', 'Female']:
        for age in age_labels:
            enter_text = f'ENTER {gender} {age}: {enter_data[gender][age]}'
            exit_text = f'EXIT {gender} {age}: {exit_data[gender][age]}'
            cvzone.putTextRect(frame, enter_text, (50, y_offset), 2, 2)
            y_offset += 40
            cvzone.putTextRect(frame, exit_text, (50, y_offset), 2, 2)
            y_offset += 40

    # Show frame
    cv2.imshow("RGB", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
