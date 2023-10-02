from keras.models import load_model
from extract_input import *


model = load_model('/Users/krishaanggupta/Desktop/ML projects/face_rec.h5')

input_path = ""
img_num = 0
vid_num = 0
while True:
    input_path = input("Please enter the pathname of the image/video for face recognition or enter \"Exit\" to end the program ")
    if input_path != "Exit":
        output_path = "/Users/krishaanggupta/Desktop/ML projects/input/"
        face_extract = input_ext(input_path, output_path, img_num, vid_num)
        print(model.predict(face_extract) + " is in the picture " + input_path)
    else:
        exit(0)  
    
    



