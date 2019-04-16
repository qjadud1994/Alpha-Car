from __future__ import division
import RPi.GPIO as GPIO
import socket,time

ip = '192.168.43.160'
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ip, 3445)
sock.connect(server_address)
print("sensor connected... port : 3445")


GPIO.setmode(GPIO.BOARD)
try :
    trig = 18
    echo = 16
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)
    print("Ultra")
    while True:
        GPIO.output(trig, False)
        time.sleep(0.5)
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        while GPIO.input(echo) == 0:
            pulse_start = time.time()
        while GPIO.input(echo) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17000
        distance = round(distance, 2)
        if (distance < 14):
            sock.send("stop".encode())
            print("Obstacle detect!")
        else:
            sock.send("gogo".encode())
        print("distance : ", distance)

except :
        GPIO.cleanup()

