import cv2
import os

def new_face():
    print("Capturing new face. Press 'q' to quit.")
    print("Please look at the camera.")
    print("Enter Your Name : ")
    name=input()
    print("Enter Your ID : ")
    id=input()

    
    cap=cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_captured=0
    if not os.path.exists('../Recognize_face_data'):
        os.makedirs('../Recognize_face_data')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame=cv2.flip(frame, 1)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
        if(face_captured==0):
            if(faces is not None):
                cv2.imwrite(f'../Recognize_face_data/{name}#{id}.jpg', frame)
                face_captured=1
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            '''face_roi = frame[y:y + h, x:x + w]
            cv2.imwrite('captured_face.jpg', face_roi)'''

        cv2.imshow('Facial Access', frame)

        if cv2.waitKey(1) == ord('q'):
            break



def detect_face():
    pass

def capture_face():
    pass







