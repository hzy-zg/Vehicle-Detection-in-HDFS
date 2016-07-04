# -*- coding: utf-8 -*-

import cv2
import sys

classifier_path = 'car_classifier.xml'
video_path = sys.argv[1]

car_cascade = cv2.CascadeClassifier(classifier_path)
cap = cv2.VideoCapture(video_path)

while True:
    ret, im = cap.read()
    if (type(im) == type(None)):
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    detected_ob = car_cascade.detectMultiScale(gray, 1.01, 1)   #Set the parameters according to requirements 

    for (x,y,w,h) in detected_ob:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)      
    
    cv2.imshow('video', im)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()