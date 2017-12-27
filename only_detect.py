import numpy as np
import cv2
import socket
from utils import interpret_netout,get_Object, Rotate, ccw
from Model2 import model
from line_object import get_lane

cap = cv2.VideoCapture(1)                   # USB 카메라로부터 영상 취득
model.load_weights("person_10_6.hdf5")         # 학습된 모델 데이터 불러오기

        # {label : [ 발견 횟수, 미발견 횟수, 발견 여부, 발견 스위치]}
Check = {'car':[0,5,False,False], 'person':[0,5,False,False], 'red':[0,5,False,False],
         'green':[0,5,False,False],'stop_sign':[0,5,False,False]}

        # 물체 이름을 코드로 변경
Encode = {'car' : 'A', 'person' : 'B', 'red' : 'C', 'green' : 'D', 'stop_sign' : 'E'}
Count = 0       # 아무것도 발견 안되는 횟수

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800,600))

    Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Gray = cv2.bitwise_not(Gray)

    input_image = cv2.resize(frame, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]               # 물체 인식을 위해
    input_image = np.expand_dims(input_image, axis=0)   # input 이미지 크기와 차원 변경
    netout = model.predict(input_image)                 # 물체 인식
    image, mark = interpret_netout(frame, netout[0])    # 물체 정보 취득

    try:
        image, left, right = get_lane(image)
    except:
        continue

    # 미발견횟수가 4이상이면 출발 신호 전송
    if Count >= 4:     # 아무것도 발견 안될때
        Count = 0
        print('start')
    if len(mark) == 0:          # 아무것도 발견 안될 때
        Count+=1                # 미발견 횟수 증가
    for m in mark:
        Obj, xmin, xmax, ymin, ymax = get_Object(image, m, Check)   # 발견 물체 정보 추출
        if Obj == None:
            Count += 1
            continue

        #print(Obj, " ", ymax)
        if Obj == 'red' or Obj == 'green' or Obj == 'stop_sign' or \
                (ccw(left, [xmax, ymax]) == 1 and ccw(right, [xmin, ymax]) == -1):
            if Check[Obj][0] >= 4:      # 발견횟수 4이상
                Check[Obj][3] = True    # 발견으로 처리
                Count = 0
                if Obj == 'green':      # 초록불 인식
                    print("restart  ", Encode[Obj] + str(ymax))
                    continue
                # 다른 물체 인식
                print("stop!  ", Encode[Obj] + str(ymax))
                break
        else:
            #print("Object : out of range")
            continue

    for m in Check:                 # 탐지 후처리
        #미발견 횟수
        if Check[m][2] == False:
            Check[m][0] = 0        #탐지 숫자 0으로q
            Check[m][1] += 1       #미탐지 숫자 증가
            if Check[m][1] >= 4 and Check[m][3] == True:
                Check[m][3] = False     # 발견 스위치 off
                print("restart")

    cv2.imshow('detect', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
