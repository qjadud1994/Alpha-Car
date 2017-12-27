import numpy as np
import copy
import cv2
from parameter import *

class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))

    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w * self.h + box.w * box.h - intersection
        return intersection / union

    def intersect(self, box):
        width = self.__overlap([self.x - self.w / 2, self.x + self.w / 2], [box.x - box.w / 2, box.x + box.w / 2])
        height = self.__overlap([self.y - self.h / 2, self.y + self.h / 2], [box.y - box.h / 2, box.y + box.h / 2])
        return width * height

    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x1


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def interpret_netout(image, netout):
    boxes = []
    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row, col, b, :5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[row, col, b, 5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].probs[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    mark = []
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        thresh = THRESHOLD
        #if LABELS[max_indx] == 'traffic_light':
        #    thresh = 0.6

        if max_prob > thresh:
            xmin = int((box.x - box.w / 2) * image.shape[1])
            xmax = int((box.x + box.w / 2) * image.shape[1])
            ymin = int((box.y - box.h / 2) * image.shape[0])
            ymax = int((box.y + box.h / 2) * image.shape[0])

            #if LABELS[max_indx] == 'traffic_light':
            #    ymin = int(ymax * 0.7)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLORS[max_indx], 5)
            cv2.putText(image, LABELS[max_indx]+" "+str(round(max_prob,2)), (xmin, ymin - 12), 0, 1e-3 * image.shape[0], (0, 255, 0), 2)
            mark.append({'label' : LABELS[max_indx], 'prob' : max_prob, 'xmin' : xmin,
                         'ymin' : ymin, 'xmax' : xmax, 'ymax' : ymax})
    return image, mark


def parse_annotation(ann_dir):
    f = open(ann_dir, 'r')
    _f = f.read()
    f_content = _f.split('\n')

    all_img = []
    current = ""

    for ann in f_content:
        img_data = ann.split(' ')
        if img_data == ['']:
            break

        file_name, width, height, xmin, ymin, xmax, ymax, label = img_data

        if not current == file_name:
            img = {'height': float(width), 'width': float(height), 'object': [], 'filename': file_name}
            current = file_name
            all_img.append(img)

        img['object'].append({'xmin': float(xmin), 'ymin': float(ymin),
                          'name': label, 'xmax': float(xmax),
                          'ymax': float(ymax)})

    return all_img


def aug_img(train_instance):
    path = train_instance['filename']
    all_obj = copy.deepcopy(train_instance['object'][:])
    img = cv2.imread(img_dir + path)
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # translate the image
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy: (offy + h), offx: (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t)

    img = img * (1 + t)
    img = img / (255. * 2.)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img[:, :, ::-1]

    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(NORM_W) / w)
            obj[attr] = max(min(obj[attr], NORM_W), 0)

        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(NORM_H) / h)
            obj[attr] = max(min(obj[attr], NORM_H), 0)

        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = NORM_W - obj['xmax']
            obj['xmax'] = NORM_W - xmin

    return img, all_obj


def data_gen(all_img, batch_size):
    num_img = len(all_img)
    shuffled_indices = np.random.permutation(np.arange(num_img))
    l_bound = 0
    r_bound = batch_size if batch_size < num_img else num_img

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_img else num_img
            shuffled_indices = np.random.permutation(np.arange(num_img))

        batch_size = r_bound - l_bound
        currt_inst = 0
        x_batch = np.zeros((batch_size, NORM_W, NORM_H, 3))
        y_batch = np.zeros((batch_size, GRID_W, GRID_H, BOX, 5 + CLASS))

        for index in shuffled_indices[l_bound:r_bound]:
            train_instance = all_img[index]

            # augment input image and fix object's position and size
            img, all_obj = aug_img(train_instance)
            # for obj in all_obj:
            #    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (1,1,0), 3)
            # plt.imshow(img); plt.show()

            # construct output from object's position and size
            for obj in all_obj:
                box = []
                center_x = .5 * (obj['xmin'] + obj['xmax'])  # xmin, xmax
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = .5 * (obj['ymin'] + obj['ymax'])  # ymin, ymax
                center_y = center_y / (float(NORM_H) / GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_indx = LABELS.index(obj['name'])
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

                    y_batch[currt_inst, grid_y, grid_x, :, 0:4] = BOX * [box]
                    y_batch[currt_inst, grid_y, grid_x, :, 4] = BOX * [1.]
                    y_batch[currt_inst, grid_y, grid_x, :, 5:] = BOX * [[0.] * CLASS]
                    y_batch[currt_inst, grid_y, grid_x, :, 5 + obj_indx] = 1.0

            # concatenate batch input from the image
            x_batch[currt_inst] = img
            currt_inst += 1

            del img, all_obj

        yield x_batch, y_batch

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_img: r_bound = num_img


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    return dst


def get_Object(image, mark, Check):
    label, prob, xmin, ymin, xmax, ymax = mark['label'], mark['prob'], mark['xmin'], \
                                          mark['ymin'], mark['xmax'], mark['ymax']
    #print("ymax : ",ymax)
    #print("label : ", label, "prob : ", prob, "xmin : ", xmin, "ymin : ", ymin, "xmax : ", xmax, "ymax : ", ymax)
    try:
        if label == 'traffic_light':
            Object = image[ymin:ymax, xmin:xmax, :]
            b, g, r = 0, 0, 0

            for y in range(ymax - ymin):
                for x in range(xmax - xmin):
                    try:
                        b += Object[y, x, 0]
                        g += Object[y, x, 1]
                        r += Object[y, x, 2]
                    except:
                        continue
            h, s, v = rgb2hsv(r,g,b)
            if h < 120:
                label = "red"
                print("red      ", h)
            elif h >= 120:
                label = "green"
                print("green    ", h)

        # 발견했을때
        Check[label][2] = True
        Check[label][0] += 1
        Check[label][1] = 0
    except:
        print("error in color extracting")
        return None, 0, 0, 0, 0

    return label, int(xmin), int(xmax), int(ymin), int(ymax)

def ccw(line, p2): # 시계반대방향알고리즘
    p0 = [line[0], line[1]]
    p1 = [line[2], line[3]]

    dx1 = p1[0] - p0[0];
    dy1 = p1[1] - p0[1];
    dx2 = p2[0] - p0[0];
    dy2 = p2[1] - p0[1];

    if (dx1 * dy2 > dy1 * dx2):
        return 1                #right
    if (dx1 * dy2 < dy1 * dx2) :
        return -1               #left

    return 0


def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v