#importing neccesary files
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import load_model


model = load_model('Forward_recognition_model.h5')

#importing neccesary functions
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
all_keypoints=[]
actions = np.array(['forward', 'backward']) 

#detection functions and drawing connections
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def extract_right_hand_points(results):
    keypoints=np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()              
    return keypoints

                
                
                
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) 
sequence = []
threshold = 0.5
output="Not resgistered Action"
pred=0

while cap.isOpened():
    ret, frame = cap.read()
    #Make detection
    image, results = mediapipe_detection(frame, hands)
    #Drawing landmarks
    draw_landmarks(image,results)
    #exctraction of keypoint
    keypoints=extract_right_hand_points(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]
    
    if len(sequence) == 30:
        
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        pred=np.max(res)
        if np.max(res)<.51:
            output="Not resgistered Action"
        else:
            output=actions[np.argmax(res)]
        print(output,pred)
    
    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image,output,(3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    cv2.imshow('OpenCV Feed', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()