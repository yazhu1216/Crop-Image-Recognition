import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Add

data_dir = '../data'
data_dir_pre = '../predict'
batch_size = 6
image_size = 224*3
random_seed = 321
split_size = 0.2
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    validation_split=split_size,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    validation_split=split_size,
    subset="validation",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)
pre_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_pre,
    label_mode="categorical",
    validation_split=0.01,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


def Position_Attention_Module(input_layer):
    gamma = tf.Variable(tf.ones(1),name="Position")
    batch_size,H, W,Channel = input_layer.shape    
    A = input_layer
    B = layers.SeparableConv2D(kernel_size=1, filters=Channel, padding='same',activation='gelu')(A)
    C = layers.SeparableConv2D(kernel_size=1, filters=Channel, padding='same',activation='gelu')(A)
    D = layers.SeparableConv2D(kernel_size=1, filters=Channel, padding='same',activation='gelu')(A)
    B = tf.transpose(B, [0, 3, 1, 2])
    C = tf.transpose(C, [0, 3, 1, 2])
    D = tf.transpose(D, [0, 3, 1, 2])
    C = tf.reshape(C, (-1, Channel, H*W))#c_n
    B = tf.transpose(tf.reshape(B, (-1, Channel, H*W)), [0, 2, 1])  # B的轉置 N_C
    beforeSoftmax = tf.matmul(B, C)
    S = tf.nn.softmax(beforeSoftmax)#N_N
    D = C#c_n
    out = tf.matmul(D, tf.transpose(S, [0, 2, 1]))#c_n
    out = tf.reshape(out, (-1, Channel, H, W))
    out = tf.transpose(out,[0,2,3,1])
    out = Add()([A, gamma*out])
    return out


def Channel_Attention_Module(input_layer):
    gamma = tf.Variable(tf.ones(1),name="Channel")
    batch_size, Channel, H, W = input_layer.shape
    A = input_layer
    A_new = tf.transpose(A, [0, 3, 1, 2])
    B = tf.reshape(A_new, (-1, Channel, H*W))
    C = tf.transpose(B, [0, 2, 1])
    beforeSoftmax = tf.matmul(B, C)
    beforeSoftmax_new = tf.reduce_max(beforeSoftmax, axis=-1, keepdims=True)
    beforeSoftmax_new = tf.repeat(beforeSoftmax_new, Channel, axis=-1)
    beforeSoftmax_new = beforeSoftmax_new - beforeSoftmax
    S = tf.nn.softmax(beforeSoftmax)
    D = B
    out = tf.matmul(S, D)
    out = tf.reshape(out, (-1, Channel, H, W))
    out = Add()([A, gamma*out])
    return out


def DANet(input_layer):
    pos = Position_Attention_Module(input_layer)
    input_layer = tf.transpose(input_layer, [0, 3, 1, 2])
    cha = Channel_Attention_Module(input_layer)
    cha = tf.transpose(cha, [0, 2, 3, 1])
    output_layer = Add()([pos, cha])
    return output_layer


def inception(x, filters):
    # 1x1
    path1 = Conv2D(filters=filters[0], kernel_size=(
        1, 1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = Conv2D(filters=filters[1][0], kernel_size=(
        1, 1), strides=1, padding='same', activation='relu')(x)
    path2 = Conv2D(filters=filters[1][1], kernel_size=(
        3, 3), strides=1, padding='same', activation='relu')(path2)

    # 1x1->5x5
    path3 = Conv2D(filters=filters[2][0], kernel_size=(
        1, 1), strides=1, padding='same', activation='relu')(x)
    path3 = Conv2D(filters=filters[2][1], kernel_size=(
        5, 5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[3], kernel_size=(
        1, 1), strides=1, padding='same', activation='relu')(path4)

    return Concatenate(axis=-1)([path1, path2, path3, path4])


def auxiliary(x, name=None):
    x = DANet(x)
    layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(x)
    layer = Conv2D(filters=128, kernel_size=(1, 1), strides=1,
                   padding='same', activation='gelu')(layer)
    layer = Flatten()(layer)
    layer = Dense(units=256, activation='gelu',
                  kernel_regularizer=regularizers.l2(0.0001))(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
    return layer


def googlenet():
    layer_in = Input(shape=IMAGE_SHAPE)
    x = data_augmentation(layer_in)
    x = layers.Rescaling(1. / 255)(x)
    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2,
                   padding='same', activation='gelu')(x)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1,
                   padding='same', activation='gelu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1,
                   padding='same', activation='gelu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    

    # Attention
    layer = DANet(layer)
    
    # stage-3
    layer = inception(layer, [64,  (96, 128), (16, 32), 32])  # 3a
    layer = inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    
    # stage-4
    layer = inception(layer, [192,  (96, 208),  (16, 48),  64])  # 4a
    aux1 = auxiliary(layer, name='aux1')
    layer = inception(layer, [160, (112, 224),  (24, 64),  64])  # 4b
    layer = inception(layer, [128, (128, 256),  (24, 64),  64])  # 4c
    layer = inception(layer, [112, (144, 288),  (32, 64),  64])  # 4d
    aux2 = auxiliary(layer, name='aux2')
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    # stage-5
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
    layer = inception(layer, [384, (192, 384), (48, 128), 128])  # 5b
    
    # layer = GlobalAveragePooling2D()(layer)
    layer = AveragePooling2D(pool_size=(
        7, 7), strides=1, padding='valid')(layer)
    
    # print(layer.shape)
    # stage-6
    layer = Flatten()(layer)
    
    # layer = tf.reshape(layer,(-1,layer.shape[-1]))
    layer = Dropout(0.5)(layer)
    layer = Dense(units=256, activation='linear',
                  kernel_regularizer=regularizers.l2(0.0001))(layer)
    main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)

    model = Model(inputs=layer_in, outputs=[main, aux1, aux2])
    # model = Model(inputs=layer_in, outputs=main)
    return model


class_names = train_ds.class_names
CLASS_NUM = len(class_names)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)
# BATCH_SIZE = 10
IMAGE_SHAPE = (image_size, image_size, 3)
MODEL_NAME = f"Googlenet_model.h5"

model = googlenet()

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint((os.path.join(
        f"Googlenet_model.h5")), monitor='aux2_accuracy', mod='max', save_best_only=True)
]
history = model.fit(train_ds, epochs=50,
                    validation_data=(val_ds), callbacks=callbacks)

model.save(MODEL_NAME)

score = model.evaluate(pre_ds)
print('Score:', score[4])
