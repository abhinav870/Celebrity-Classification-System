import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil
import pywt

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

# plt.imshow(face_img)
# plt.show()

# """
# DETECTING THE EYES IN THE FACE
# """
# cv2.destroyAllWindows()
#
# # iterating over all the faces present in our image
# for (x, y, w, h) in faces:
#
#     # drawing rectangle around the face of the image
#     face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     # print("Printing Face image Vector: ",face_img) # this is a 3-D vector
#
#     # extract the pixels representing the brightness of gray image from gray array
#     roi_gray = gray[y:y + h, x:x + w]
#
#     # extract the pixels representing the brightness of original image from face_img array
#     roi_color = face_img[y:y + h, x:x + w]
#
#     # using the loaded eyes features we get a 2-D eyes vector
#     # The vector comprises of ex,ey,ew,eh of all the eyes present in a particular face
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#
#     # iterating over all the eyes in a particular face
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        # drawing green colour rectangle around the eyes

# printing the detected eyes in our image
# plt.figure()
# plt.imshow(face_img, cmap='gray')
# plt.show()

#  we are interested in facial region of every image in our dataset. So we will be cropping the face region from all the
# images. We will store these cropped images into a different folder and use it for our model training
# plt.imshow(roi_color, cmap='gray')
# plt.show()

# A function where we input the original image and the function returns us the cropped face , if the face and 2 eyes
# are detected clearly. We will run this function on all of our images

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

# If path of cropped folder exists, then we delete it.
# This is because next time we run this we want to perform data cleaning to remove unwanted images from there
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
else:
    os.mkdir(path_to_cr_data)# Otherwise we create it

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