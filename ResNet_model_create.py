import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Add,Conv2D
from tensorflow.keras.applications import ResNet50,ResNet152
from tensorflow.keras.activations import gelu
from tensorflow import keras
import numpy as np


class ResnetModel:
    def __init__(self, num_classes, input_shape, resnet, data_aug=False, weights="imagenet", flatten=False, dual=False,resnet_input = 448):
        self.resnet_input = resnet_input
        self.num_classes = num_classes
        # 圖像數據增強
        self.data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ]
        )
        self.input_shape = input_shape
        self.data_aug = data_aug
        self.resnet = resnet
        self.weights = weights
        self.flatten = flatten
        self.dual = dual

    def Position_Attention(self, input_layer):
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

    def Channel_Attention(self, input_layer):
        gamma = tf.Variable(tf.ones(1),name="Channel")
        _, C, H, W = input_layer.shape
        origin_input = input_layer
        transform_input = tf.reshape(input_layer, (-1, C, H * W))
        B = tf.transpose(transform_input, [0, 2, 1])
        C_B = tf.matmul(transform_input, B)
        C_B = tf.nn.softmax(C_B)
        C_B_D = tf.matmul(C_B, transform_input)
        C_B_D_reshape = tf.reshape(C_B_D, (-1, C, H, W))
        output_layer = Add()([origin_input, gamma*C_B_D_reshape])
        return output_layer

    def DANet(self, input_layer):
        pos = self.Position_Attention(input_layer)
        input_layer = tf.transpose(input_layer, [0, 3, 1, 2])
        cha = self.Channel_Attention(input_layer)
        cha = tf.transpose(cha, [0, 2, 3, 1])
        output_layer = Add()([pos, cha])
        return output_layer

    def Resnet_build(self, ):
        inputs = Input(shape=self.input_shape, name="Data_aug")
        if self.data_aug:
            x = self.data_augmentation(inputs)
            x = layers.Rescaling(1. / 255)(x)
        else:
            x = layers.Rescaling(1. / 255)(inputs)

        resnet_model = self.resnet(include_top=False, weights=self.weights, input_tensor=x, pooling='avg',
                                   classifier_activation='softmax')

        model = tf.keras.Model(resnet_model.inputs,
                               Dense(self.num_classes, activation="softmax", name="output_layer")(resnet_model.output))
        if self.dual:
            input_layer = resnet_model.layers[81].output
            dual_attention_layers = self.DANet(input_layer)
            # attention_model = tf.keras.Model(resnet_model.inputs,
            #                        Dense(self.num_classes, activation="softmax", name="output_layer")(dual_attention_layers))
            # dual_attention_layers = i(dual_attention_layers)
            # resnet_model.layers[82].input = dual_attention_layers
            resnet_model.layers[82](dual_attention_layers)
            model = tf.keras.Model(resnet_model.inputs,
                                   Dense(self.num_classes, activation="softmax", name="output_layer")(resnet_model.layers[-1]))

        if self.flatten:
            x = model.layers[-3].output
            x = Flatten(name='flatten')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation=gelu)(x)
            x = Dropout(0.5)(x)
            output_layer = Dense(self.num_classes, activation="softmax", name="output_layer")(x)
            model = tf.keras.Model(model.input, output_layer)
        print(model.layers[81].output.shape)
        return model

if __name__ == "__main__":
    image_size = 224
    resnet = ResnetModel(num_classes=14, input_shape=(image_size, image_size, 3), resnet=ResNet50,
                     data_aug=False,weights=None,flatten=False,dual=True)
    model = resnet.Resnet_build()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="binary_crossentropy", metrics=['accuracy'])
    # model.summary()
