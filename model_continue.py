from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Add
import tensorflow as tf
import numpy as np
import json
import os

floder = './1400file/mixmodel/5'
model = tf.keras.models.load_model(floder+'/'+'Googlenet_model.h5')
data_dir = 'data3'
print(model.layers)
# model.layers[1].trainable = True
# model.layers[2].trainable = True
# inputs = tf.keras.layers.Input(shape=(model1.input.shape[1:])) 
# model1._name = 'ZhuCute'
# model1.trainable = False

# model1.layers.pop(0)
# layers1 = model1(inputs)
# addlayer = Add()([layers1[0],layers1[-1]])
# outlayers = Dense(14,activation='softmax')(addlayer)
# model = tf.keras.Model(inputs,outlayers)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),loss="categorical_crossentropy", metrics=['accuracy'])


# with open((os.path.join(floder,'params.json')),'r') as f:
#             model_params = json.load(f)
        
# intput_shape = model_paramsz.get('input_shape')
# Aug = model_params.get('Aug')
# Weights = model_params.get('Weights')
# Faltten = model_params.get('Faltten')
# Dual = model_params.get('Dual')

split_size = 0.2
random_seed = 123
# image_size = intput_shape[0]
image_size = 224*3
batch_size = 3

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    validation_split=split_size,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    validation_split=split_size,
    subset="validation",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size)
pre_ds = tf.keras.utils.image_dataset_from_directory(
    'predict',
    label_mode="categorical",
    validation_split=0.1,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size)
class_names = train_ds.class_names
train_ds = train_ds.prefetch(buffer_size=42)
val_ds = val_ds.prefetch(buffer_size=42)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint((os.path.join(floder,"best_2_{epoch}.h5")),monitor='val_accuracy',mod='max',save_best_only=True),
    # tf.keras.callbacks.ModelCheckpoint((os.path.join(floder,"Recall2.h5")),monitor='val_recall',mod='max',save_best_only=True),
    # tf.keras.callbacks.ModelCheckpoint((os.path.join(floder,"Precision2.h5")),monitor='val_precision',mod='max',save_best_only=True)
]
epochs = 20
history = model.fit(
    train_ds, epochs=epochs,callbacks=callbacks,validation_data=val_ds,
)


np.save((floder+'/history2.npy'),history.history)
model.evaluate(pre_ds)
model.save(floder+'/model1000.h5')