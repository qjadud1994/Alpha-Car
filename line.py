import cv2
import numpy as np
import time

global lx1, lx2, ly1, ly2, gx1, gx2, gy1, gy2, warped_eq1
lx1, lx2, ly1, ly2, gx1, gy1, gx2, gy2 = int(0), int(0), int(0), int(0),int(0), int(0), int(0), int(0)


# Frame width & Height
w = 800
h = 600

def ROI(img, vertices, color3=(255, 255, 255), color1=255):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        color = color3
    else:
        color = color1

    cv2.fillPoly(mask, vertices, color)
    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image

def get_fitline_left(img, f_lines):
    global lx1, lx2, ly1, ly2
    lines = f_lines.reshape(f_lines.shape[0] * 2, 2)
    vx, vy, x, y = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)

    if lines.shape[0] != 0:
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
        lx1, ly1, lx2, ly2 = x1, y1, x2, y2

    else:
        if lx1 == 0 and lx2 == 0 and ly1 == 0 and ly2 == 0:
            x1, y1 = img.shape[1] / 2, img.shape[0] / 2
            x2, y2 = img.shape[1] / 2, img.shape[0] / 2
        else:
            x1, y1, x2, y2 = lx1, ly1, lx2, ly2

    result = [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2]
    return result

def get_fitline_right(img, f_lines):
    global gx1, gx2, gy1, gy2
    lines = f_lines.reshape(f_lines.shape[0] * 2, 2)
    vx, vy, x, y = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)

    if lines.shape[0] != 0:
        x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
        x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
        gx1, gy1, gx2, gy2 = x1, y1, x2, y2

    else:
        if gx1 == 0 and gx2 == 0 and gy1 == 0 and gy2 == 0:
            x1, y1 = img.shape[1] / 2, img.shape[0] / 2
            x2, y2 = img.shape[1] / 2, img.shape[0] / 2
        else:
            x1, y1, x2, y2 = gx1, gy1, gx2, gy2

    result = [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2]
    return result

def draw(img, lines):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), (0, 0, 255), 10)


count=0
def get_lane(img):

    #readRefImages()
    global line

    height, weight = img.shape[:2]

    #canny_img = cv2.Canny(Gray, 50, 200)  # black, white
    canny_img = cv2.Canny(img, 150, 200)#black, white


    #cv2.imshow("canny", canny_img)

    vertices = np.array([[(0, height), (0, height / 2), (weight, height / 2), (weight, height)]],
                        dtype=np.int32)

    #vertices = np.array([[(0, height), (100, height / 2), (weight-100, height / 2), (weight, height)]],
    #                    dtype=np.int32)

    ROI_img = ROI(canny_img, vertices)
    cv2.imshow('roi', ROI_img)
    img = cv2.polylines(img, vertices, True, (255, 0 ,0), 5)
    line_arr = cv2.HoughLinesP(ROI_img, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=50)

    if type(line_arr).__name__ == 'NoneType':
        line_arr = line
    elif line_arr.shape[0] != 1:
        line_arr = np.squeeze(line_arr)
        line = line_arr
    elif line_arr.shape[0] == 1:
        line_arr = np.squeeze(line_arr, axis=1)
        line = line_arr

    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    line_arr1,line_arr2,slope_degree1,slope_degree2 = line_arr,line_arr,slope_degree,slope_degree
    line_arr1 = line_arr1[np.abs(slope_degree1) < 180]
    slope_degree1 = slope_degree1[np.abs(slope_degree1) < 180]
    line_arr1 = line_arr1[np.abs(slope_degree1) > 90]
    slope_degree1 = slope_degree1[np.abs(slope_degree1) >90]

    line_arr2 = line_arr2[np.abs(slope_degree2) > 0]
    slope_degree2 = slope_degree2[np.abs(slope_degree2) > 0]
    line_arr2 = line_arr2[np.abs(slope_degree2) < 90]
    slope_degree2 = slope_degree2[np.abs(slope_degree2) < 90]

    if line_arr1.shape[0]!=0 and line_arr2.shape[0]!=0:
        line_arr = np.concatenate((line_arr1,line_arr1),axis=0)
        slope_degree = np.concatenate((slope_degree2,slope_degree1),axis=0)

    elif line_arr1.shape[0]!=0 and line_arr2.shape[0]==0:

        line_arr = line_arr1
        slope_degree =  slope_degree1
    elif line_arr1.shape[0]==0 and line_arr2.shape[0]!=0:
        line_arr = line_arr2
        slope_degree = slope_degree2

    else:

        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]
        line_arr = line_arr[np.abs(slope_degree) >30]
        slope_degree = slope_degree[np.abs(slope_degree)>30]

    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]

    if L_lines.shape[0]!=0 and R_lines.shape[0]==0:
        #vertices = np.array([[(0, height), (0, height * 2 / 3), (weight/2, height * 2 / 3), (weight/2, height)]],
        #                    dtype=np.int32)
        vertices = np.array([[(0, height), (0, height / 2), (weight / 2, height / 2), (weight / 2, height)]],
                            dtype=np.int32)
        ROI_img = ROI(canny_img, vertices)
        line_arr = cv2.HoughLinesP(ROI_img, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=50)
        if type(line_arr).__name__ == 'NoneType':
            line_arr = line
        elif line_arr.shape[0] != 1:
            line_arr = np.squeeze(line_arr)
            line = line_arr
        elif line_arr.shape[0] == 1:
            line_arr = np.squeeze(line_arr, axis=1)
            line = line_arr

        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        line_arr1, line_arr2, slope_degree1, slope_degree2 = line_arr, line_arr, slope_degree, slope_degree

        line_arr1 = line_arr1[np.abs(slope_degree1) < 160]
        slope_degree1 = slope_degree1[np.abs(slope_degree1) < 160]
        line_arr1 = line_arr1[np.abs(slope_degree1) > 90]
        slope_degree1 = slope_degree1[np.abs(slope_degree1) > 90]

        line_arr2 = line_arr2[np.abs(slope_degree2) > 30]
        slope_degree2 = slope_degree2[np.abs(slope_degree2) > 30]
        line_arr2 = line_arr2[np.abs(slope_degree2) < 90]
        slope_degree2 = slope_degree2[np.abs(slope_degree2) < 90]

        if line_arr1.shape[0] != 0 and line_arr2.shape[0] != 0:
            line_arr = np.concatenate((line_arr1, line_arr1), axis=0)
            slope_degree = np.concatenate((slope_degree2, slope_degree1), axis=0)

        elif line_arr1.shape[0] != 0 and line_arr2.shape[0] == 0:
            line_arr = line_arr1
            slope_degree = slope_degree1
        elif line_arr1.shape[0] == 0 and line_arr2.shape[0] != 0:
            line_arr = line_arr2
            slope_degree = slope_degree2
        else:
            line_arr = line_arr[np.abs(slope_degree) < 160]
            slope_degree = slope_degree[np.abs(slope_degree) < 160]
            line_arr = line_arr[np.abs(slope_degree) > 30]
            slope_degree = slope_degree[np.abs(slope_degree) > 30]

        L_lines, R_lines = line_arr[(slope_degree > 0), :], np.array([[weight,height, weight, height*0.7]])

    elif L_lines.shape[0]==0 and R_lines.shape[0]!=0:
        #vertices = np.array([[(weight/2, height * 2 / 3), (weight/2, height), (weight, height * 2 / 3), (weight, height)]],
        #                    dtype=np.int32)
        vertices = np.array([[(0, height), (0, height / 2), (weight / 2, height / 2), (weight / 2, height)]],
                            dtype=np.int32)

        ROI_img = ROI(canny_img, vertices)
        line_arr = cv2.HoughLinesP(ROI_img, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=50)
        if type(line_arr).__name__ == 'NoneType':
            line_arr = line
        elif line_arr.shape[0] != 1:
            line_arr = np.squeeze(line_arr)
            line = line_arr
        elif line_arr.shape[0] == 1:
            line_arr = np.squeeze(line_arr, axis=1)
            line = line_arr

        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        line_arr1, line_arr2, slope_degree1, slope_degree2 = line_arr, line_arr, slope_degree, slope_degree

        line_arr1 = line_arr1[np.abs(slope_degree1) < 160]
        slope_degree1 = slope_degree1[np.abs(slope_degree1) < 160]
        line_arr1 = line_arr1[np.abs(slope_degree1) > 90]
        slope_degree1 = slope_degree1[np.abs(slope_degree1) > 90]

        line_arr2 = line_arr2[np.abs(slope_degree2) > 30]
        slope_degree2 = slope_degree2[np.abs(slope_degree2) > 30]
        line_arr2 = line_arr2[np.abs(slope_degree2) < 90]
        slope_degree2 = slope_degree2[np.abs(slope_degree2) < 90]

        if line_arr1.shape[0] != 0 and line_arr2.shape[0] != 0:
            line_arr = np.concatenate((line_arr1, line_arr1), axis=0)
            slope_degree = np.concatenate((slope_degree2, slope_degree1), axis=0)

        elif line_arr1.shape[0] != 0 and line_arr2.shape[0] == 0:
            line_arr = line_arr1
            slope_degree = slope_degree1
        elif line_arr1.shape[0] == 0 and line_arr2.shape[0] != 0:
            line_arr = line_arr2
            slope_degree = slope_degree2
        else:
            line_arr = line_arr[np.abs(slope_degree) < 160]
            slope_degree = slope_degree[np.abs(slope_degree) < 160]
            line_arr = line_arr[np.abs(slope_degree) > 30]
            slope_degree = slope_degree[np.abs(slope_degree) > 30]

        L_lines, R_lines = np.array([[0, height, 0, height*0.7]]), line_arr[(slope_degree < 0), :]


    #temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    left_fit_line = get_fitline_left(img, L_lines)
    right_fit_line = get_fitline_right(img, R_lines)

    draw(img, right_fit_line[:4])
    draw(img, right_fit_line[:4])
    draw(img, left_fit_line[:4])
    center_x_point = int((left_fit_line[4] + right_fit_line[4]) / 2)
    center_y_point = int((left_fit_line[5] + right_fit_line[5]) / 2)
    result = cv2.circle(img, (center_x_point, center_y_point), 5, (0, 0, 255), -1)

    #result = cv2.addWeighted(img, 1, temp, 1, 0)

    center_x_point = str(center_x_point)
    if len(center_x_point) == 1:
        center_x_point = "000"+center_x_point
    elif len(center_x_point) == 2:
        center_x_point = "00"+center_x_point
    elif len(center_x_point) == 3:
        center_x_point = "0"+center_x_point

    return result, center_x_point
