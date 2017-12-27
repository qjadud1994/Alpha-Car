import numpy as np
import cv2
import socket
import argparse
from utils import interpret_netout,get_Object, Rotate
from Model2 import model

parser = argparse.ArgumentParser()
parser.add_argument('method', type=str)
parser.add_argument('path', type=str)

args = parser.parse_args()
method = args.method
test_path = args.path

cap = cv2.VideoCapture(test_path)
model.load_weights(method)

while cap.isOpened():
    ret, frame = cap.read()
    input_image = cv2.resize(frame, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, axis=0)

    netout = model.predict(input_image)
    image, mark = interpret_netout(frame, netout[0])

    image = cv2.resize(image, (1024, 768))
    cv2.imshow('detect', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
