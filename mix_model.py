from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Add
import tensorflow as tf
import os
import numpy as np
import shutil
import json

data_dir = './data'
data_dir_pre = './predict'
batch_size = 7
image_size = 224*3
random_seed = 321
split_size = 0.1
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
    validation_split=0.001,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)

model1 = tf.keras.models.load_model('./Googlenet_model.h5')
model2 = tf.keras.models.load_model('./best_ResNet_model.h5')
inputs = tf.keras.layers.Input(shape=(model1.input.shape[-3:]))
model1.layers.pop(0)
model2.layers.pop(0)
model1.trainable = False
model2.trainable = False
layers1 = model1(inputs)
layers2 = model2(inputs)
concatlayer = tf.concat([layers1[0],layers2],1)
addlayer = Add()([layers1[0],layers2])
addlayer = Dense(14,activation='gelu')(addlayer)
addlayer = Dropout(0.5)(addlayer)
concatlayer = Dense(28,activation='gelu')(concatlayer)
concatlayer = Dropout(0.5)(concatlayer)
concatlayer = Dense(28,activation='gelu')(concatlayer)
concatlayer = Dropout(0.5)(concatlayer)
concatlayer = Dense(14,activation='softmax')(concatlayer)
addlayer = Add()([addlayer,concatlayer])
outlayers = Dense(14,activation='softmax')(addlayer)
model  = tf.keras.Model(inputs,outlayers)

floder_model = '1400file/mixmodel'
save_floder = os.path.join(floder_model,str(len(os.listdir(floder_model))))
if not os.path.exists(save_floder):
    os.mkdir(save_floder)

learning_rate = 0.005
epochs = 10
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss="categorical_crossentropy", metrics=['accuracy'])

try:
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint((os.path.join(save_floder,f"best_mix_model.h5")),monitor='val_accuracy',mod='max',save_best_only=True),
    ]
    history = model.fit(
        train_ds, epochs=epochs,callbacks=callbacks,validation_data=val_ds
    )
    np.save((os.path.join(save_floder,f'history.npy')),history.history)
    # model.evaluate(pre_ds)
    model.save((os.path.join(save_floder,f'model.h5')))
except:
    if("best.h5" not in os.listdir(save_floder)):
        shutil.rmtree(save_floder,ignore_errors=True)