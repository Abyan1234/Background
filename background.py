import cv2
import time
import numpy as np


fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))


cap=cv2.VideoCapture(0)


time.sleep(2)
bg=0


for i in range(60):
    ret,bg=cap.read()

#Flipping the background
bg=np.flip(bg,axis=1)

#Reading the captured frame until the camera is open
while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    
    #Flipping the image for consistancy
    img=np.flip(img,axis=1)

    #converting the color from rgb to hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Generating mask to detect red color
    #This value can aslo be changed to suite the color
    lower_red=np.array([0,120,50])
    upper_red=np.array([210,255,255])
    mask_1=cv2.inRange(hsv,lower_red,upper_red)

    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])
    mask_2=cv2.inRange(hsv,lower_red,upper_red)
    mask_1=mask_1+mask_2

    #Open and expand the image where there is mask 1
    #morphalogyEx(src,dst,op,kernel)
    #Src=Source of the input image: Dst=Destination(Output Image):Op=Integer represents type of the morphalogical operation:Kernel=Metrics
    mask_1=cv2.morphalogyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1=cv2.morphalogyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    #Selecting only the part that doesn't have mask 1 and saving in mask 2
    mask_2=cv2.bitwise_not(mask_1)

    #Keeping only the part of the images without the red color or any other color you may choose
    res_1=cv2.bitwise_and(img,img,mask=mask_2)
    res_2=cv2.bitwise_and(bg,bg,mask=mask_1)

    #Generate the final output by combining res_1 and res_2
    final_ouput=cv2.addWeighted(res_1,1,res_2,1,0)
    out.write(final_ouput)
    cv2.imshow("magic",final_ouput)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyallwindow()