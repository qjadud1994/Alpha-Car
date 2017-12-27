from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adagrad
import numpy as np
import tensorflow as tf

from parameter import *
from utils import parse_annotation, data_gen
from Model2 import model

all_img = parse_annotation(ann_dir)


def custom_loss(y_true, y_pred):
    ### Adjust prediction
    # adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)

    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])

    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)

    ### Adjust ground truth
    # adjust x and y
    center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W) / GRID_W), (float(NORM_H) / GRID_H)], [1, 1, 1, 1, 2])
    true_box_xy = center_xy - tf.floor(center_xy)

    # adjust w and h
    true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1, 1, 1, 1, 2]))

    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
    true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh

    intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

    # adjust confidence
    true_box_prob = y_true[:, :, :, :, 5:]

    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    # y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor

    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf

    weight_prob = tf.concat(CLASS * [true_box_conf], 4)
    weight_prob = SCALE_PROB * weight_prob

    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)

    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W * GRID_H * BOX * (4 + 1 + CLASS)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss


layer = model.layers[-3] # the last convolutional layer
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

try:
    model.load_weights("deepcoco_weights3.hdf5")
    print("Previous data")
except:
    print("New data")


sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9)
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=7, mode='min', verbose=1)
checkpoint = ModelCheckpoint('deepcoco_weights3.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)

#sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9)

model.compile(loss=custom_loss, optimizer=sgd)
model.fit_generator(generator=data_gen(all_img, BATCH_SIZE),
                    steps_per_epoch=int(len(all_img)/BATCH_SIZE),
                    epochs = 100,
                    verbose = 1,
                    callbacks = [early_stop, checkpoint],
                    max_queue_size = 3)

