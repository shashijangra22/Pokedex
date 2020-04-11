from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="Path to trained model")
ap.add_argument("-l","--labelbin",required=True,help="path to label binarizer")
ap.add_argument("-i","--image",required=True,help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output = image.copy()

image = cv2.resize(image,(96,96))
image = image.astype("float")/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"],"rb").read())

print("[INFO] classifying image...")
probs = model.predict(image)[0]
idx = np.argmax(probs)
label = lb.classes_[idx]

filename = args["image"][args["image"].rfind(os.path.sep)+1:]

text = "{}: {:.2f}%".format(label, probs[idx]*100)
output = imutils.resize(output, width=400)
cv2.putText(output,text,(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

print("[INFO] {}".format(text))
cv2.imwrite("o"+filename,output)