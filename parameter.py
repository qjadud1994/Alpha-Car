#LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'traffic_light', 'street_sign', 'stop_sign', 'parking_meter']
#LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic_light', 'stop_sign']
LABELS = ['person', 'car', 'traffic_light', 'stop_sign']
WIDTH = 800
HEIGHT = 600

COLORS = [(43,206,72),(255,204,153),(120,150,0),(148,255,181)]

input_shape = (416, 416, 3)
NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13 , 13
BATCH_SIZE = 8
BOX = 5
CLASS = len(LABELS)
THRESHOLD = 0.65
ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
