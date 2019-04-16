import cv2
import socket
from line import get_lane
from utils import Rotate

width, height = 800, 600
ip = '192.168.43.160'

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ip, 3442)
sock.connect(server_address)

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture(1)
print(cap.isOpened())

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (width, height))

    #Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Gray = cv2.bitwise_not(Gray)

    try:
        frame, point = get_lane(frame)
        print(point)
        sock.send(point.encode())
    except:
        print("error")
        pass

    cv2.imshow('a', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
