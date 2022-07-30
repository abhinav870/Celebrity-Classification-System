import joblib
import json
import numpy as np
import base64
import cv2
from Wavelet_Transform import w2d

# private variables
__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def classify_image(image_base64_data, file_path=None):

    # If we have clear face and 2 eyes, then it will be returned and contained in imgs
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []

    # Iterating over images one by one
    for img in imgs:

        # rescaling our images , applying wavelet transform
        # and then vertically stacking wavelet transform image and raw images
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)


        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

"""
Takes input as a number returns the name of the celebrity
"""

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

"""
Loading Saved Model and Dictionary

"""


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    # Loading Saved Model
    with open("C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\artifacts\\class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    # Loading Saved Dictionary
    global __model
    if __model is None:
        with open('C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\artifacts\\saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# this function is taking base64 string and returning us cv2 image
def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\opencv\\haarcascades\\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\opencv\\haarcascades\\haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces


# "b64.txt"--> virat kohli binary64 encoded image
# "base64_test.txt"----> serena williams binary encoded image
def get_b64_test_image():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    print(classify_image(get_b64_test_image(), None))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\federer1.jpg"))

    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\virat1.jpg"))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\virat2.jpg"))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\virat3.jpg"))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\serena1.jpg"))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\serena2.jpg"))
    print(classify_image(None, "C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\sharapova1.jpg"))

    # C:\\Users\\GARG\\PycharmProjects\\Python Codes\\Image Classification Project\\server\\test_images\\