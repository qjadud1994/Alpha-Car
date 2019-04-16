# Alpha-Car

### Member
- 이용한 : 팀장, 차선 인식 
- 김범영 : 딥러닝, 물체 인식
- 이지훈 : Motor, Sensor 제어
- 이신효 : 통신

### What is Alpha-Car?
- 2017 한이음 공모전에 나간 Alpha-car팀의 '딥러닝 기반 자율 주행 버스 운행 시스템' 이다.
- Deep Learning, OpenCV, Raspberry pi, RFID, 초음파 센서 등을 이용하여 구현하였다.


### Implement Details
- SW
  - lane tracing using OpenCV (image processing Library)
  - Object Detection using Deep Learning ([YOLO v2](https://arxiv.org/pdf/1612.08242.pdf))
  - We can detect Car, Pedestrian, Stop sign and Traffic sign.

- HW
  - DC Motor for driving power
  - Servo Motor for direction control
  - Ultrasonic sensor for front obstacle detection
  - RFID for Bus Stop Recognition

### Result
- 2017 한이음 공모전 금상(과기정통부 장관상) 수상 프로젝트이며 자세한 사항들은 아래 링크에서 확인할 수 있다.
- [Youtube 데모 영상](https://www.youtube.com/watch?v=BcBvTIv5zpw&t=1s)
- [프로젝트 보고서 - 한이음 수상작 페이지](http://www.hanium.or.kr/portal/project/awardList.do)
