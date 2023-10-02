# using this to modify the images into usable dataset for training the model
# 
# code this to extract faces from all the photos and 
# then make a folder of all the extracted photos

import cv2
import glob

face_class = cv2.CascadeClassifier('/Users/krishaanggupta/Desktop/ML projects/haarcascade_frontalface_default.xml')

path_train = "/Users/krishaanggupta/Desktop/ML projects/trainingdata/Unknown/*.*"
path_valid = "/Users/krishaanggupta/Desktop/ML projects/validationdata/Unknown/*.*"

img_list = glob.glob(path_valid, recursive = True)
img_num = 1

for file in img_list:
    #print(file)
    img = cv2.imread(file, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(gray, 1.3, 5)
    try: 
        for (x,y,w,h) in faces:
            roi_color = img[y:y+h, x:x+w]
        resized = cv2.resize(roi_color, (224,224))
        cv2.imwrite("/Users/krishaanggupta/Desktop/ML projects/validationext/Unknown/"+str(img_num)+".jpg", resized)
    except:
        print("No faces detected")
        #resized = cv2.resize(img, (224,224))
        #cv2.imwrite("/Users/krishaanggupta/Desktop/ML projects/trainingext/Sarika/"+str(img_num)+".jpg", resized)
        
    img_num += 1 
    
        