import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height=image.shape[0]
    #here polygon is triangle
    polygon=np.array([[(200,height),(1100,height),(550,250)]])#############BUG
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    #to show cropped image(lane image is cropped according to mask image)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def make_coordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=int(image.shape[0])#height of image
    y2=int(y1*(3/5))#############BUG
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])


def average_slope_intercept(image,lines):
    left_fit=[]# coordinates of left lines
    right_fit=[]# coordinates of right lines
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        # slope of left lines will always be negative which can be clearly seen using matplot image of combo image
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))

    if len(left_fit) and len(right_fit):
    ##over-simplified if statement (should give you an idea of why the error occurs)
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines


def display_lines(image,lines):
    #creating a new blank image
    line_image=np.zeros_like(image)
    #lines return coordinates x1,y1,x2,y2
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line
            #writing line coordinates one by one in new black image(line_image)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image





# image=cv2.imread('test_imageJ.jpeg')
# lane_image=np.copy(image)
# canny_img=canny(lane_image)
# cropped_image=region_of_interest(canny_img)

# lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)#?????
# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_lines(lane_image,averaged_lines)
# #to superimpose lane_image with line_image
# combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)# we mul with 0.8 to decrease pixel intensity
# #cv2.imshow('result',combo_image)
#
# cv2.waitKey(0)


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        canny_image = canny(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
