from __future__ import division
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import socket, threading, time

ip = '172.20.10.3'

Encode = {'A' : 400, 'B' : 330, 'C' : 200, 'D' : 200, 'E' : 150}
        #    car    /   person /   red    /   green  /  stop_sign
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(50)

front_left = 12
front_right = 13
back_left = 15
back_right = 11

GPIO.setmode(GPIO.BOARD)
GPIO.setup(front_left, GPIO.OUT)
GPIO.setup(front_right, GPIO.OUT)
GPIO.setup(back_left, GPIO.OUT)
GPIO.setup(back_right, GPIO.OUT)

CAR_CENTER = 410
CENTER = 307
center = CENTER
SPEED = 1400
CUR_SPEED = 1400
pwm.set_pwm(0, 0, CENTER)

def straight(pwm=pwm, GPIO=GPIO, speed=CUR_SPEED, center=CENTER):
    pwm.set_pwm(0, 0, center)
    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)
    GPIO.output(front_left, GPIO.HIGH)
    GPIO.output(front_right, GPIO.HIGH)
    GPIO.output(back_left, GPIO.LOW)
    GPIO.output(back_right, GPIO.LOW)

def stop(pwm=pwm, GPIO=GPIO, speed=CUR_SPEED):
    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)
    GPIO.output(front_left, GPIO.HIGH)
    GPIO.output(front_right, GPIO.HIGH)
    GPIO.output(back_left, GPIO.LOW)
    GPIO.output(back_right, GPIO.LOW)

def left(pwm=pwm, GPIO=GPIO, speed=CUR_SPEED, center=CENTER, turn=100):
    pwm.set_pwm(0, 0, center + turn)  # 410
    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)
    GPIO.output(front_left, GPIO.HIGH)
    GPIO.output(front_right, GPIO.HIGH)
    GPIO.output(back_left, GPIO.LOW)
    GPIO.output(back_right, GPIO.LOW)

def right(pwm=pwm, GPIO=GPIO, speed=CUR_SPEED, center=CENTER, turn=100):
    pwm.set_pwm(0, 0, center - turn)  # 210
    pwm.set_pwm(4, 0, speed)
    pwm.set_pwm(5, 0, speed)
    GPIO.output(front_left, GPIO.HIGH)
    GPIO.output(front_right, GPIO.HIGH)
    GPIO.output(back_left, GPIO.LOW)
    GPIO.output(back_right, GPIO.LOW)

Object = False
Bus_stop = False
Obstacle = False

def from_line():
    global Object, Bus_stop, Obstacle
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, 3442)
    print("line socket listening...")
    sock.bind(server_address)
    sock.listen(1)

    try:
        client, address = sock.accept()
        print("Line Connected")
        while True:
            data = client.recv(4)
            if Object == True or Bus_stop == True or Obstacle == True:
                print("stop")
                stop(speed=0)
                continue
            try:
                x_point = int(data)
            except:
                continue
            if x_point >= CAR_CENTER - 50 and x_point <= CAR_CENTER + 50:
                print("straight : ", x_point)
                straight()
            elif CAR_CENTER >= x_point:
                k = int((CAR_CENTER - x_point) / 6)
                print("right : ", x_point," -> ",k + CENTER)
                right(speed=CUR_SPEED + 50, turn=k)
            elif CAR_CENTER < x_point:
                k = int((x_point - CAR_CENTER) / 6)
                print("left : ", x_point, " -> ", k + CENTER)
                left(speed=CUR_SPEED + 50, turn=k)
    except:
        print("close line")
        GPIO.cleanup()
        exit(0)


def from_YOLO():
    global Object, Bus_stop, Obstacle, CUR_SPEED
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, 3443)
    print("YOLO socket listening...")
    sock.bind(server_address)
    sock.listen(1)
    try:
        client, address = sock.accept()
        print("YOLO connected")
        while True:
            data = client.recv(4)
            if data == 'gogo':
                if not Bus_stop and not Obstacle:
                    stop(speed=SPEED)
                    Object = False
                continue

            Type, ymax = data[0], int(data[1:])
            if Type == 0 or Type == None or Type == '0':
                continue

            if Encode[Type] < ymax:
                stop(speed=0)
                Object = True
                print("Object : Stop!!")
            elif Encode[Type] - 300 >= ymax:
                CUR_SPEED = 1400
                print("Object : enough distance")
            else:
                CUR_SPEED = Encode[Type] - ymax + 1100  #1100 is min-speed
                print("Object : Slow Down ",CUR_SPEED)

    except:
        print("close yolo")
        sock.close()
        GPIO.cleanup()
        exit(1)

def from_RFID():
    global Bus_stop, Obstacle
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, 3444)
    print("Sensor socket listening...")
    sock.bind(server_address)
    sock.listen(1)

    try:
        client, address = sock.accept()
        while True:
            data = client.recv(4)
            if data == "gogo" and not Obstacle:
                stop(speed=SPEED)
                Bus_stop = False
            elif data == "stop":
                stop(speed=0)
                Bus_stop = True
    except:
        print("close RFID")
        sock.close()
        GPIO.cleanup()
        exit(1)

def from_Ultra():
    global Obstacle, Bus_stop
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, 3445)
    print("Sensor socket listening...")
    sock.bind(server_address)
    sock.listen(1)

    try:
        client, address = sock.accept()
        while True:
            data = client.recv(4)
            if data == "gogo" and not Bus_stop:
                stop(speed=SPEED)
                Obstacle = False
            elif data == "stop":
                stop(speed=0)
                Obstacle = True
    except:
        print("close ultra")
        sock.close()
        GPIO.cleanup()
        exit(1)

LINE = threading.Thread(target=from_line)
DETECT = threading.Thread(target=from_YOLO)
RFID = threading.Thread(target=from_RFID)
ULTRA = threading.Thread(target=from_Ultra)

LINE.start()
DETECT.start()
RFID.start()
ULTRA.start()

LINE.join()
DETECT.join()
RFID.join()
ULTRA.join()

pwm.set_pwm(0, 0, 0)
GPIO.cleanup()
