import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

X = []

padding = 10
skip = 0

#print(sys.argv) 

while True:

    # read the images
    ret, frame =cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame)
    #print(faces)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 2)

        if skip%10 == 0:
            face_section = frame[y-padding:y+h+padding, x-padding:x+w+padding]
            face_section = cv2.resize(face_section, (150, 150))
            #print(face_section.shape)
            X.append(face_section.flatten())
            print(len(X))
        skip += 1

        cv2.imshow("Frame captured", frame)
        cv2.imshow('Face Section', face_section)

    key_pressed = cv2.waitKey(1) &0xFF
    #print(key_pressed)
    if key_pressed == ord('q'):
        print(chr(key_pressed), key_pressed)
        break 

X = np.array(X)
print(X.shape)

filename = input("Enter the name of the person")
np.save(filename, X)

cap.release()
cv2.destroyAllWindows()


