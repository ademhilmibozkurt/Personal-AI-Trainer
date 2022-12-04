# this modules purpose is use estimator in other projects

import cv2 as cv
import mediapipe as mp
import numpy as np
import time


class PoseDetector():
    
 
    def __init__(self, mode=False, upperBody=False, smooth=True,
                 detectionCon=True, trackCon=0.5):
        
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody,
                                     self.smooth, self.detectionCon, self.trackCon)
        
    def findPose(self, img, draw=True):
    
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks:
            if draw:    
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
        return img

    def getPosition(self, img, draw=True):
        
        self.lmList = []
        if self.results.pose_landmarks:
            
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])    
                
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 0), cv.FILLED)
           
        return self.lmList
    
    
    def getAngle(self, img, p1, p2, p3, draw=True):
       
       x1, y1 = self.lmList[p1][1:]
       x2, y2 = self.lmList[p2][1:]
       x3, y3 = self.lmList[p3][1:]
       
       #angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y2-y1, x2-x1))
       radians = np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2)
       angle = np.abs(radians*180.0/np.pi)
       
       if angle >180.0:
           angle = 360-angle
       
       if draw:
           cv.line(img, (x1, y1), (x2, y2), (200, 53, 20), 3)
           cv.line(img, (x2, y2), (x3, y3), (200, 53, 20), 3)
           
           cv.circle(img, (x1, y1), 8, (53, 200, 20), cv.FILLED)
           cv.circle(img, (x1, y1), 14, (53, 200, 20), 2)
           cv.circle(img, (x2, y2), 8, (53, 200, 20), cv.FILLED)
           cv.circle(img, (x2, y2), 14, (53, 200, 20), 2)
           cv.circle(img, (x3, y3), 8, (53, 200, 20), cv.FILLED)
           cv.circle(img, (x3, y3), 14, (53, 200, 20), 2)
           cv.putText(img, str(int(angle)), (x2+30, y2), cv.FONT_HERSHEY_PLAIN, 2, (53, 200, 20), 2)
  
  
def main(capture):
    
    pTime=0
    detector = PoseDetector()  
    
    while True:
       
        success, img = capture.read()
        img = detector.findPose(img)
    
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    
        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_PLAIN,
                3, (20, 53, 200), 3)
    
        cv.imshow('Poses',img)
        if cv.waitKey(1) & 0xFF==ord(" "):
            break
    

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='create capture')
    parser.add_argument('--capture', metaver='path', required=True, help='path to capture')
    arg = parser.parse_args()
    main(capture=arg.capture)
    
    
    
    