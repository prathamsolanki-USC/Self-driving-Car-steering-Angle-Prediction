
import cv2 as cv
import numpy as np
import imutils
import easyocr
import pyperclip
from selenium import webdriver
import time
import pyautogui
import pdfkit

image='car3gg.jfif'
#READ IMAGE,grayscale
img=cv.imread(image)
print(img)
cv.imshow('carorignal',img)
cv.waitKey(0)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('car_GRAY',gray)
#cv.waitKey(0)

#apply filter and fine edge for localization
bfilter=cv.bilateralFilter(gray,11,17,17)#noise reduction
#cv.imshow('car_Bilateral Filtering',bfilter)
#cv.waitKey(0)

edged=cv.Canny(bfilter,30,200)
#cv.imshow('car_Edge Detection',edged)
#cv.waitKey(0)


#contours
keypoints=cv.findContours(edged.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

contours=imutils.grab_contours(keypoints)
contours=sorted(contours,key=cv.contourArea,reverse=True)[:10]
location=None
for contour in contours:
    approx=cv.approxPolyDP(contour,10,True)
    if len(approx)==4:
        location=approx
        break

print("Location of number plate on image is")
print(location)

#masking
mask=np.zeros(gray.shape,np.uint8)
new_image=cv.drawContours(mask,[location],0,255,-1)
new_image=cv.bitwise_and(img,img,mask=mask)

new_image2=cv.cvtColor(new_image,cv.COLOR_BGR2RGB)
# cv.imshow('car_Mask',new_image2)
# cv.waitKey(0)


#ocr

(x,y)=np.where(mask==255)#finding every section which is not black
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
cropped_image= gray[x1:x2+1,y1:y2+1]
# cv.imshow('car_ocr',cropped_image)
# cv.waitKey(0)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#render ocr image
reader=easyocr.Reader(['en'])
result=reader.readtext(cropped_image)
print("OCR reuslt is")
print(result)


#manu
if result[0][1]=='IND':
    result=result[1]
    print("new result is")
    print(result)
    text=result[1]

elif len(result)==2 and result[0][1]!='IND':
    ans=result[0][1]+result[1][1]
    text=ans



else:
    #text=numberplate
    text=result[0][1]


#print("text is")
#print(text)

#manupulation

#to remove space
if ' ' in text:
    def removespace(string):
        return string.replace(" ", "")

    # Driver Program
    string = text
    n = removespace(string)
    #print(n)

elif '.' in text:
    def removespace(string):
        return string.replace(".", "")
    #Driver Program
    string = text
    n = removespace(string)
    print(n)

else:
    n=text



#actual manu
a = list(n[2:4])
b = []
for i in a:
    if i != int:
        if i == 'Z':
            i = 2
            b.append(i)
        elif i == 'I':
            i = 1
            b.append(i)
        elif i == 'G':
            i = 6
            b.append(i)
        elif i == 'S':
            i = 5
            b.append(i)
        elif i == 'B':
            i = 8
            b.append(i)

        elif i == 'O':
            i = 0
            b.append(i)

        elif i == '0':
            i = '0'
            b.append(i)

        elif i == '1':
            i = 1
            b.append(i)

        elif i == '2':
            i = 2
            b.append(i)

        elif i == '3':
            i = 3
            b.append(i)

        elif i == '4':
            i = 4
            b.append(i)

        elif i == '5':
            i = 5
            b.append(i)

        elif i == '6':
            i = 6
            b.append(i)

        elif i == '7':
            i = 7
            b.append(i)

        elif i == '8':
            i = 8
            b.append(i)
        elif i == '9':
            i = 9
            b.append(i)

m = list(n)


m[2] = b[0]
m[3] = b[1]

s = m
# list to string
number_plate = ''.join([str(elem) for elem in s])


text=number_plate



#put text on image
clipb_copy=pyperclip.copy(text)
print('the car number plate reads :{}'.format(text))

font=cv.FONT_HERSHEY_SIMPLEX
res=cv.putText(img,text=text,org=(approx[0][0][0],approx[1][0][1]+60),fontFace=font,fontScale=1,color=(0,255,0),thickness=2)
res=cv.rectangle(img,tuple(approx[0][0]),tuple(approx[2][0]),(0,255,0),3)
# cv.imshow('carfinal',res)
# cv.waitKey(0)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# to print state
state=text[0:2]

dict={'AP':'Andhra Pradesh','AR':'Arunachal Pradesh','AS':'Assam',
      'BR':'Bihar','CH':'Chandigarh','CT':'Chhattisgarh','DN':'Dadra and Nagar Haveli','DD':'Daman and Diu',
      'DL':'Delhi','GA':'Goa',"GJ":"gujurat",'HR':'Haryana','HM':'Himachal Pradesh','JK':'Jammu and Kashmir',
      'JH':'Jharkhand','KA':'Karnataka','KL':'Kerala','LD':'Lakshadweep','MP':'Madhya Pradesh','MH':'Maharashtra',
      'MN':'Manipur','ML':'Meghalaya','MZ':'Mizoram','NL':'Nagaland','OR':'Odisha',
      'PY':'Puducherry','PB':'Punjab','RJ':'rajasthan','SK':'Sikkim','TN':'Tamil nadu','TG':'Telangana',
      'TR':'Tripura','UP':'Uttar pradesh','UT':'uttarakhand','WB':'West Bengal'
}

for x,y in dict.items():
    if x==state:
        print('car registration is of {}'.format(y))
        break

cv.destroyAllWindows()


#UI automation
driver=webdriver.Chrome('C:\\Users\\Pratham\\slenium\\chromedriver.exe')
# driver=webdriver.Chrome('C:\\Users\\Pratham\\selenium\\chromedriver.exe')

driver.get("https://vahan.nic.in/nrservices/faces/user/citizen/citizenlogin.xhtml")


driver.find_element_by_name('TfMOBILENO').send_keys('9820668924')
driver.find_element_by_name('btRtoLogin').click()
driver.find_element_by_name('tfPASSWORD').send_keys('password@123')
driver.find_element_by_name('btRtoLogin').click()
driver.find_element_by_name('regn_no1_exact').send_keys(text)
driver.save_screenshot('D:\Pratham\Desktop\sss\img.png')
# cv.waitKey(5000)
time.sleep(8)
pyautogui.click(x=962,y=757)
time.sleep(1)
pyautogui.click(x=668,y=793)
time.sleep(3)

driver.execute_script('window.print();')
# driver.find_element_by_xpath('//*[@id="sidebar"]//print-preview-button-strip//div/cr-button[1]').click()





