import os
import cv2
import pytesseract
import re
import numpy as np
from pytesseract import Output
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def ocrtt(input_img):
    image = cv2.imread(input_img)

    # get grayscale image
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # template matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    gray = get_grayscale(image)
    thresh = thresholding(gray)
    opening = opening(gray)
    canny = canny(gray)
    deskew = deskew(gray)

    text0 = pytesseract.image_to_string(image)
    text = pytesseract.image_to_string(gray)
    text1 = pytesseract.image_to_string(thresh)
    text2 = pytesseract.image_to_string(opening)
    text3 = pytesseract.image_to_string(deskew)

    textf = text0 + text + text1 + text2 + text3

    def findDob(textf):
        try:
            DOB = re.findall(r'\d{2}\/\d{2}\/\d{4}', textf)[0]
        except:
            try:
                DOB = re.findall(r'Year of Birth|DOB[\:\s\d\/]+', textf)[0].split()[-1]
            except:
                DOB = ''
        return (DOB)

    def findGender(textf):
        wlist = re.findall('[A-Za-z]{4,6}', textf)
        wlist = list(map(lambda x: x.lower(), wlist))
        if ('female' in wlist):
            gender = 'FEMALE'
        elif 'male' in wlist:
            gender = 'MALE'
        else:
            gender = 'NOT FOUND'
        return (gender)

    gender = findGender(textf)

    dob = findDob(textf)

    aadhar_number = re.findall(r'\S{4}\s\S{4}\s\d{4}', textf)[0]

    aadhar_name = re.findall(r'[A-z][a-z]{1,} [A-Z][a-z]{1,}\s*[a-zA-Z]*', textf)[1]

    aadhar_details = [aadhar_name, aadhar_number, dob, gender]

    return aadhar_details






@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    global filename
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)

    aadhar_det = ocrtt(destination)

    name = aadhar_det[0]
    aadh_num = aadhar_det[1]
    dob = aadhar_det[2]
    gender = aadhar_det[3]

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename,name=name,aadh_num=aadh_num,dob=dob,gender=gender)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == '__main__':
    app.run(debug=True)
