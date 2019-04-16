from __future__ import division
import RPi.GPIO as GPIO
import socket, time
import MFRC522
from multiprocessing import Process, Queue


ip = '192.168.43.160'
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ip, 3444)
sock.connect(server_address)
print("sensor connected... port : 3444")

MIFAREReader = MFRC522.MFRC522()
print("RFID")
while True:
    (status, TagType) = MIFAREReader.MFRC522_Request(MIFAREReader.PICC_REQIDL)
    if status == MIFAREReader.MI_OK:
        print("STOP")
        sock.send("stop".encode())
        time.sleep(5)
        sock.send("gogo".encode())
        time.sleep(5)

GPIO.cleanup()



