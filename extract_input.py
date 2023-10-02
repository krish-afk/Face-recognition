# check if input image or video:

#if image just detect face and run the model on it;
#if video convert the video into folder of photos and then run the model
#on it to detect all the faces in it


 
import cv2
import glob
import os

input_path = input("Please put the pathname of the image/ video: ")

face_class = cv2.CascadeClassifier('/Users/krishaanggupta/Desktop/ML projects/haarcascade_frontalface_default.xml')

img_num = 0
vid_num = 0
output_path = "/Users/krishaanggupta/Desktop/ML projects/input/"

def input_ext(input_path, output_path, img_num, vid_num):
    image_extensions = ['.jpg', '.jpeg']
    vid_extension = '.mp4'

    img_bool = False
    vid_bool = False
    fold_bool = False

    for ext in image_extensions:
        if ext in input_path:
            img_bool = True
    if vid_extension in input_path:
        vid_bool = True
    if (img_bool is False) and (vid_bool is False) and ("." not in input_path):
        fold_bool = True
    elif (img_bool is False) and (vid_bool is False) and ("." in input_path):
        print("Unsupported datatype, please try again")
        return
    
    if fold_bool:
        img_list = glob.glob(input_path, recursive = True)
        img_num = 0
        out_path = input_path + "ext"
        try: 
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        except OSError:
            print("Error creating a directory")
        for img in img_list:
            return input_ext(img, out_path, img_num, vid_num)
        return
    
    if img_bool:
        img_num += 1
        img = cv2.imread(input_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_class.detectMultiScale(gray, 1.3, 5)
        try: 
            for (x,y,w,h) in faces:
                roi_color = img[y:y+h, x:x+w]
            resized = cv2.resize(roi_color, (224,224))
            img_path = output_path + "image" + str(img_num) + ".jpg"
            cv2.imwrite(img_path, resized)
        except:
            print("No faces detected")
        return img_path
    
    elif vid_bool:
        vid_num += 1
        vid = cv2.VideoCapture(input_path)
        try: 
            data_path = "/Users/krishaanggupta/Desktop/ML projects/input/vid" + str(vid_num)
            out_path = data_path + "ext"
            if not os.path.exists(data_path):
               os.makedirs(data_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
        except OSError:
            print("Error creating directory for video data")
        
        frame_num = 0
        while(True):
            ret,frame = vid.read()
            if ret:
                pic_name =  data_path + "/frame" + str(frame_num) + ".jpg"
                cv2.imwrite(pic_name, frame)
                frame_num += 1
            else:
                break
    
        vid.release()
        
        return input_ext(data_path, out_path, img_num, vid_num, img_bool, vid_bool)    
