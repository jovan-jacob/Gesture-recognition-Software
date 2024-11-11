import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
mphands=mp.solutions.hands
hands = mphands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

mpDraw=mp.solutions.drawing_utils
pTime=0
cTime=0

inpt=1
while inpt!=0:
    inpt=int(input())
    success, img= cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # for id,lm in enumerate(handLms.landmark):
            #     print(id,lm)
            #     h,w,c=img.shape
            #     cx,cy=int(lm.x*y),int(lm.y*h)
            #     print(id,cx,cy)
            #     if id==0:
            #         cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            
            mpDraw.draw_landmarks(img,handLms, mphands.HAND_CONNECTIONS)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(0,0), cv2.FONT_HERSHEY_COMPLEX,3,(255,8,255),3)        
    
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)

