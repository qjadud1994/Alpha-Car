from keras.models import Model
from keras.layers import Reshape, Conv2D, MaxPooling2D, Lambda
from keras.layers import BatchNormalization, Input, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

from parameter import CLASS, BOX, input_shape

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)

def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def Conv_BN_LR_MP(inputs, depth, pooling=True):
    x = Conv2D(depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    if pooling:
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x

def Conv_bottleneck(inputs, count, side_depth, middle_depth, pooling=True):
    x = Conv2D(side_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(middle_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(side_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    if count == 5 or count == 7:
        x = Conv2D(middle_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(side_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    if count == 7:
        x = Conv2D(side_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(side_depth, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    if pooling:
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x


inputs = Input(shape=input_shape)

conv1 = Conv_BN_LR_MP(inputs, 32)
conv2 = Conv_BN_LR_MP(conv1, 64)
conv3_5 = Conv_bottleneck(conv2, count=3, side_depth=128, middle_depth=64)
conv6_8 = Conv_bottleneck(conv3_5, count=3, side_depth=256, middle_depth=128)
conv9_13 = Conv_bottleneck(conv6_8, count=5, side_depth=512, middle_depth=256, pooling=False)

conv21 = conv9_13

conv9_13 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv9_13)
conv14_20 = Conv_bottleneck(conv9_13, count=7, side_depth=1024, middle_depth=512, pooling=False)

conv21 = Conv_BN_LR_MP(conv21, 64, pooling=False)
conv21 = Lambda(
        space_to_depth_x2,
        output_shape=space_to_depth_x2_output_shape,
        name='space_to_depth')(conv21)

merge = concatenate([conv14_20, conv21])
conv22 = Conv_BN_LR_MP(merge, 1024, pooling=False)
conv23 = Conv2D(5*(CLASS+BOX), (1, 1), strides=(1, 1))(conv22)

conv23 = Activation('linear')(conv23)
final = Reshape((13, 13, 5, CLASS+BOX))(conv23)

model = Model(inputs, final)
model.summary()
