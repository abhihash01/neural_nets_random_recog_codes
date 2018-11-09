import cv2
import os
import subprocess
from time import sleep
from multiprocessing import Process

cascPath = 'lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
cam = cv2.VideoCapture(0)

cv2.namedWindow("check")
data = 'abhilash'
img_counter = 0
path = '/media/abhi/362A9DAA2A9D681F/ext/imagecap1/checkimages'





with open('output.txt', 'r') as myfile:
  data = myfile.read()
flag = 0

while True:
    path = '/media/abhi/362A9DAA2A9D681F/ext/imagecap1/checkimages'
    name = raw_input("press enter and start capturing")
    if not os.path.exists(os.path.join(path, name)):
     os.makedirs(os.path.join(path, name))
     path = os.path.join(path, name)

    while True:
        ret, frame = cam.read()
        #cv2.imshow("test", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
             gray,
             scaleFactor=1.1,
             minNeighbors=5,
             minSize=(30,30)

        )
        #or use faces
        for (x, y, w, h) in faces:
         if x != 0:
          flag=1
          img_name = "opencv_frame_{}.png".format(img_counter)
          cv2.imwrite(os.path.join(path, img_name), frame)
          cv2.waitKey(0)
          print("{} written!".format(img_name))
          img_counter += 1
          print("transfer call initiated")
          subprocess.call(['./transfer1.sh'])
          print("transfer call returned")
          with open('output.txt', 'r') as myfile:
           data = myfile.read()
          #flag = 0


        #the rectangles are bound by the boxes that tract ehe x and w along with the h of the image, once the face is detected
        #this is what makes the automation of faces that once comes into the field of view of the image
      
        for(x, y, w, h) in faces
        print("as the point where the face was first detecte")
        
        #display the resulting frame
        cv2.imshow('cam', frame)
        #if x == 0:
        #  flag = 0
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
         # ESC pressed
            print("Escape hit, closing...")
            break
        #elif k%256 == 32:
         # SPACE pressed
        elif k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(os.path.join(path, img_name), frame)
            cv2.waitKey(0)
            print("{} written!".format(img_name))
            img_counter += 1
    p = raw_input("to continue or not")
    if p == '1':
      cam.release()
      print("training images gathered")
      break    
        



cam.release()

cv2.destroyAllWindows()
print("transfer call initiated")
subprocess.call(['./transfer1.sh'])
print("transfer call returned")
with open('output.txt', 'r') as myfile:
  data = myfile.read()
