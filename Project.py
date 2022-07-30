import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil
import pywt

"""
(1) Preprocessing: Detect face and eyes
When we look at any image, most of the time we identify a person using a face.
An image might contain multiple faces, also the face can be obstructed and not clear.
The first step in our pre-processing pipeline is to detect faces from an image.
Once face is detected, we will detect eyes, if two eyes are detected then only we keep that image otherwise discard it
"""

path = r'C:\\Users/GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\test_images\\sharapova1.jpg'

img = cv2.imread(path) # reading the image from the path
print("Printing Image Shape ",img.shape)
# the third dimension in the image shape indicates that the image comprises of 3 RGB channels

# plt.imshow(img)
# plt.show()
# plotting the image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Printing Gray image shape ",gray.shape)
# converts the image from RGB to grayscale

# print("Printing the Gray Vector ",gray)
# an N-D array representing the brightness of each pixel of our image (in the range of 0-255 arranged in an array)

# plt.imshow(gray, cmap='gray')
# plt.show()
# plotting the grayscale image

# loads the face features and eyes features
face_cascade = cv2.CascadeClassifier('C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\opencv\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\opencv\\haarcascades\\haarcascade_eye.xml')

# detects the face using face features (just like masks in CNN)
# the faces array will have 4 values representing : x coordiante, y-coordiante, width, height of each face in our image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
print("Printing the coordinates of face ",faces)

# getting the 4 values
(x,y,w,h) = faces[0]
print("Printing the 4 features of the first face in our image: ",x,y,w,h)

face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# (x,y) = beginning indices of rectangle around the face
# (x+w,y+h) = ending indices of rectangle around the face
# we are drawing a rectangle of red colour(255,0,0) with beg and ending indices as shown above

plt.imshow(face_img)
plt.show()

""" 
DETECTING THE EYES IN THE FACE 
"""
cv2.destroyAllWindows()

# iterating over all the faces present in our image
for (x, y, w, h) in faces:

    # drawing rectangle around the face of the image
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # print("Printing Face image Vector: ",face_img) # this is a 3-D vector

    # extract the pixels representing the brightness of gray image from gray array
    roi_gray = gray[y:y + h, x:x + w]

    # extract the pixels representing the brightness of original image from face_img array
    roi_color = face_img[y:y + h, x:x + w]

    # using the loaded eyes features we get a 2-D eyes vector
    # The vector comprises of ex,ey,ew,eh of all the eyes present in a particular face
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # iterating over all the eyes in a particular face
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # drawing green colour rectangle around the eyes

# printing the detected eyes in our image
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()

"""
(2) Crop the facial region of the image

"""

plt.imshow(roi_color, cmap='gray')
plt.show()
cropped_img = np.array(roi_color)
print(cropped_img.shape)

"""

3 (a) Preprocessing: Use wavelet transform as a feature for training our model
In wavelet transformed image, you can see edges clearly and that can give us clues on various facial features 
such as eyes, nose, lips 

"""

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

im_har = w2d(cropped_img,'db1',5)
plt.imshow(im_har, cmap='gray')
plt.show()

"""
You can see above a wavelet transformed image that gives clues on facial features such as eyes, nose, lips etc.
This along with raw pixel image can be used as an input for our classifier

3(b) Preprocessing: Load image, detect face. If eyes >=2, then save and crop the face region
Lets write a python function that can take input image and returns cropped image (if face and eyes >=2 are detected)

"""

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # if number of eyes detected in the face are equal to 2, then we return the region of interest
        if len(eyes) >= 2:
            return roi_color

original_image = cv2.imread(path)
# plt.imshow(original_image)
# plt.show()

cropped_img = get_cropped_image_if_2_eyes(path)
# plt.imshow(cropped_img)
# plt.show()

# In case the 2 eyes aren't clearly visible, our function doesn't return a cropped image of the facial region
path2 = r'C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\test_images\\sharapova2.jpg'

# plotting the obstructed image
# org_image_obstructed = cv2.imread(path2)
# plt.imshow(org_image_obstructed)
# plt.show()

# Calling the function for an obstructed image
cropped_image_no_2_eyes = get_cropped_image_if_2_eyes(path2)
print(cropped_image_no_2_eyes)

"""
Above cropped_image_no_2_eyes is None which means we should 
ignore this image and we will not use such image for model training

"""

# Creating new folder in images_dataset directory
path_to_data = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\dataset"
path_to_cr_data = "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\dataset\\cropped"

img_dirs = [] # empty list to store path of individual sub folders in images_dataset

# os.scandir() it goes through all the sub-directories (5 in our case: Messi,Sharapova,Federer,Williams, VK)
# within the images_dataset folder
for entry in os.scandir(path_to_data):
    # isdir() method in Python is used to check whether the specified path is an existing directory or not.
    # If the specified path is a symbolic link pointing to a directory then the method will return True else False
    # so if the given sub folder in images_dataset directory exists i.e. its path is valid, we add it to the img_dirs
    if entry.is_dir():
        img_dirs.append(entry.path)

print("Printing Directory of all the Sub-Folders inside dataset directory :",img_dirs)

"""
Go through all images in dataset folder and create cropped images for them.
There will be cropped folder inside dataset folder after you run this code

"""

# If path of cropped folder exists, then we delete it.
# This is because next time we run this we want to perform data cleaning to remove unwanted images from there
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
else:
    os.mkdir(path_to_cr_data) # Otherwise we create it

cropped_image_dirs = []
celebrity_file_names_dict = {}

# iterating through directories of all the Sub-Folders inside images_dataset directory
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('\\')[-1]
    print(celebrity_name)

    celebrity_file_names_dict[celebrity_name] = []

    # iterate through each of these 5 folders and all the images in each of these folders
    for entry in os.scandir(img_dir):

        # get the roi_color of the image being processed
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None: # means eyes were >= 2

            # we create the cropped_folder for a particular celebrity
            cropped_folder = path_to_cr_data + "\\" + celebrity_name
            if not os.path.exists(cropped_folder): # if path doesn't exist we create a directory
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)

            cropped_file_name = celebrity_name + str(count) + ".png" # eg virat_kohli12.png
            cropped_file_path = cropped_folder + "\\" + cropped_file_name

            # cv2.imwrite() method is used to save an image to any storage device.
            # This will save the image according to the specified format in current working directory
            # cv2.imwrite(filename, image)
            # filename: A string representing the file name. The filename must include image format like .jpg, .png
            # image: It is the image that is to be saved.
            cv2.imwrite(cropped_file_path, roi_color)

            # creating a dictionary where key is celebrity name and values are path of cropped images of the celebrity
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1

# example:
# {
#   'lionel_messi': [
#                       './datasets/cropped/messi/messi1.png',
#                       './datasets/cropped/messi/messi2.png',
#                   ],

#   'virat_kohli':  [
#                       './datasets/cropped/kohli/kohli1.png'
#                       './datasets/cropped/kohli/kohli2.png'
#                   ],
# }

"""

(4) Now you should have cropped folder under datasets folder that contains cropped images¶
    Manually examine cropped folder and delete any unwanted image
    
"""


"""

(5) Vertically stacking raw image and wavelet transform image for all the images lying in our dataset

"""

# creating 5 different classes for our classifier
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count += 1
print("Printing Class_dict",class_dict)


X, y = [], []

# first loop will iterate through all the celebrity names  in celebrity_file_names_dict
# second loop will iterate through every image for that particular celebrity
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:

        img = cv2.imread(training_image)  # reading the image
        if img is None:
            continue
        scaled_raw_img = cv2.resize(img, (32, 32)) # resizing the raw image
        img_har = w2d(img,'db1',5) # getting wavelet transform image
        scaled_img_har = cv2.resize(img_har, (32, 32)) # resizing the wavelet transform image
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        # vertically stacking these images (*3 because this is a coloured image- RGB channels)

        X.append(combined_img)
        y.append(class_dict[celebrity_name])

print("Printing the total number of images" ,len(X))
print("Printing the Size of each image ",len(X[0])) # 32*32*3(raw image) + 32*32 (wavelet transform)

X = np.array(X).reshape(len(X),4096).astype(float) # X has new dimensions as 162 * 4096 ans its contents are float 
print("Printing Modified Dimensions of X",X.shape)

