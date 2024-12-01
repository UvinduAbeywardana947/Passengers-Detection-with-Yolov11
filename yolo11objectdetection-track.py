import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import time

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load the YOLO11 model
model = YOLO("yolo11s.pt")
names=model.model.names
# Open the video file (use video file or webcam, here using webcam)
cap = cv2.VideoCapture('people1.avi')
count=0
cy1=390
cy2=400
offset=8
inp={}
enter=[]
exp={}
exit=[]

# Timer for 15-minute intervals
start_time = time.time()
interval_duration = 15 * 60  # 15 minutes in seconds

while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,classes=0)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx= int(x1+x2)//2
            cy= int(y1+y2)//2
         
        # Check crossing lines for entering 
            if cy1<(cy+offset) and cy1<(cy-offset): 
                inp[track_id]= (cx,cy)
            if track_id in inp:
                if cy2<(cy+offset) and cy2<(cy-offset):
                    cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    cvzone.putTextRect(frame,f'{track_id}',(x1,y2),1,1)
                    cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
                    if enter.count(track_id) == 0:
                        enter.append(track_id)
            
        # Check the crossing lines for exiting
            if cy2<(cy+offset) and cy2<(cy-offset): 
                exp[track_id] = (cx, cy)
            if track_id in exp:
                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    cv2.circle(frame, (cx, cy), 4, (255,0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    if exit.count(track_id) == 0:
                       exit.append(track_id)

    cv2.line(frame,(270,400),(900,400),(0,255,0),2)
    cv2.line(frame,(268,390),(900,390),(255,255,0),2)
    enterp=len(enter)
    cvzone.putTextRect(frame,f'ENTERPERSON:{enterp}',(50,60),2,2)
    exitp = len(exit)
    cvzone.putTextRect(frame, f'EXITPERSON:{exitp}', (50,100), 2, 2)


    cv2.imshow("RGB", frame)
    # Check if the 15-minute interval has elapsed
    if time.time() - start_time > interval_duration:
        # Display counts of enter and exit passengers
        print(f"Passengers Entered in 15 minutes: {len(enter)}")
        print(f"Passengers Exited in 15 minutes: {len(exitp)}")
        
        # Reset for the next interval
        start_time = time.time()
        enter.clear()
        exitp.clear()

    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

