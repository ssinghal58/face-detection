import cv2
import numpy as np

# loading cascade xmls
face_cascade = cv2.CascadeClassifier()
eye_cascade = cv2.CascadeClassifier()
face_cascade.load('haarcascade_frontalface_default.xml')
eye_cascade.load('haarcascade_eye.xml')
# starting webcam
cap = cv2.VideoCapture(0)

while True:
    # reading captured frames
    ret,frame = cap.read()
    # grayscale conversion
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # list of detected faces
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        # drawing ractangle on boundries detected
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0),2)
        # extracting face
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        # detecting eyes from extracted face area
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # drawing rectangle on boundaries detected for eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0),2)
    # showing final image
    cv2.imshow('img',frame)
    # wait until user press q or ctrl+z
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# release camera
cap.release()
# Destroying all windows
cv2.destroyAllWindows()