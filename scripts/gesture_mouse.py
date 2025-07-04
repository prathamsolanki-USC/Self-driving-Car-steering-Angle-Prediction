import scipy.signal as signal
import matplotlib.pyplot as plt
from tkinter import *
# hand tracking
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import os
import smtplib
import pyautogui

def start():
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction
    smoothening = 7

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    # print(wScr, hScr)
    hold = False

    while True:
        # time.sleep(0.1)
        #Find hand Landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            x5, y5 = lmList[16][1:]#ring finger
            # print(x1, y1, x2, y2)

            # Check which fingers are up
            fingers = detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)
            if fingers[1] == 1 and fingers[2] == 0:
            #Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                #Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                autopy.mouse.move(wScr - clocX, clocY)

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY


            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                # # cv2.circle(img, (x5, y5), 15, (0, 0, 0), cv2.FILLED)
                #
                # if length < 40:
                #     cv2.circle(img, (lineInfo[4], lineInfo[5]),
                #                15, (0, 255, 0), cv2.FILLED)
                #     autopy.mouse.click()
                #
                #     startx=x1
                #     starty=y1
                #     endx,endy=x1,y1
                if fingers[3]==1:
                    hold=True
                elif fingers[3]==0:
                    hold=False
                if length<40:
                    pyautogui.mouseDown()
                    if hold==False:
                        pyautogui.mouseUp()
                    # pyautogui.moveTo(x1,y1)
                    # if fingers[3] == 1 and hold != True:
                    #     print("yesss")
                    #     cv2.circle(img, (x5, y5), 15, (0, 0, 0), cv2.FILLED)
                    #     hold = True
                    #     pyautogui.mouseDown(duration=1)
                    #
                    # if fingers[3] == 0 :
                    #     pyautogui.mouseUp()
                    #
                    #     if hold==True:
                    #         hold=False
                pyautogui.moveTo(x1,y1)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

start()