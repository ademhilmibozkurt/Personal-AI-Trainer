import cv2 as cv
from module import PoseEstimationModule as pm

# read image
#img = cv.imread('source/3.jpg')

# read video from 'source'
capture = cv.VideoCapture("source/0.mp4")

# send capture to pm module
pm.main(capture)

# read video from camera
#capture = cv.VideoCapture(0)
    
# create detector instance
detector = pm.PoseDetector(capture)

# variables
stage = ""
counter = 0

# loop for show video and catch frames
while True:
    # read frame
    success, img = capture.read()
    
    # resize image from video
    #img = cv.resize(img, (int(img.shape[1]/1.5), int(img.shape[0]/1.5)))
    
    # resize image from photo
    #img = cv.resize(img, (int(img.shape[1]/1.5), int(img.shape[0]/1.5)))
    
    # resize image from camera (if you need)
    #img = cv.resize(img, (int(img.shape[1]*2), int(img.shape[0]*2)))
    
    # find landmark nodes coordinates
    detector.findPose(img, False)
    lmList = detector.getPosition(img, False)
    
    if lmList != 0:
        angle = detector.getAngle(img, 11, 13, 15, True)
      
    # curl counter
    if angle >150:
        stage = "down"
    if angle <40 and stage=="down":
        stage = "up"
        counter +=1
    
    # write stage and count on screen
    # status box
    cv.rectangle(img, (0,0), (250, 73), (200, 53, 20), -1)
    
    # reputations
    cv.putText(img, 'Reps', (10,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(img, str(counter), (10,65), cv.FONT_HERSHEY_SIMPLEX, 2, (255,200,200), 2, cv.LINE_AA)
    
    # stages
    cv.putText(img, 'Stage', (110,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    cv.putText(img, stage, (95,65), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,200,200), 2, cv.LINE_AA)
    
    
    # show videos
    cv.imshow("Trainer", img)
    
    # if pressed space in window pause video
    # waitkey(1) keep window open
    if cv.waitKey(1) & 0xFF == ord(" "):
        break

capture.release()
cv.destroyAllWindows()    