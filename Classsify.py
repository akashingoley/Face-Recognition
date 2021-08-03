import numpy as np
import cv2
import os 

X = []
y=[]
class_id = 0
d = {}

l = os.listdir('./')
for fname in l:
    if fname.endswith('.npy'):
        data = np.load(fname)
        print(fname, data.shape)
        X.append(data)

        labels = class_id*np.ones((data.shape[0], 1))
        y.append(labels)
        
        d[class_id] = fname.split('.')[0]
        class_id += 1

print(d)
X = np.concatenate(X, axis=0)
print(X.shape)
y = np.concatenate(y, axis=0)
print(X.shape)

#CLASSIFIER

def predict(X, y, test_point, k=10):
    # compute distance
    distance = ((X - test_point)**2).sum(axis=1)**0.5
    knn_indexes = np.argsort(distance)[:k] # indices of k nearest neighbour

    # get the category of my NN
    knn_cat = y[knn_indexes]
    cls, count = np.unique(knn_cat, return_counts=True)
    

    max_count_idx = np.argmax(count)
    pred_cat = cls[max_count_idx] # category with maximum count

    return pred_cat

def accuracy(ypred, ytrue):
    return (ypred==ytrue).mean()


#TEST POINT

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

padding = 10
skip = 0

while True:

    # read the frame
    ret, frame = cap.read()
    if ret==False:
       continue 

    faces = face_cascade.detectMultiScale(frame)
    for face in faces:
        x, y_, w, h = face
        if skip%10 == 0:
            face_section = frame[y_-padding:y_+h+padding,  x-padding:x+w+padding]
            face_section = cv2.resize(face_section, (150,150)) # test sample should have same size as training point
            
            label = predict(X, y, face_section.flatten())
            predname = d[label]

        skip += 1
    
    cv2.rectangle(frame, (x,y_), (x+w, y_+h), (0, 0,255), 2)
    cv2.putText(frame, predname, (x,y_), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
    cv2.imshow("Frame captured", frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()