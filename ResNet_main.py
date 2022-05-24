from model_create import ResnetModel
import tensorflow as tf
from tensorflow.keras.applications import ResNet50,ResNet152
import numpy as np
import os
import shutil
import json
from tensorflow.keras.metrics import Recall,Precision
import tensorflow_addons as tfa
data_dir0 = 'data'

image_size = 224 * 3
batch_size = 3


model_input = ResNet152
Dual = True
Faltten = False
Aug = False
Weights = None
learning_rate = 0.0001
weight_decay = 0.00001

random_seed = 321
split_size = 0.2

train_ds0 = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir0,
    label_mode="categorical",
    validation_split=split_size,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size
)
val_ds0 = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir0,
    label_mode="categorical",
    validation_split=split_size,
    subset="validation",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size)
pre_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'predict',
    label_mode="categorical",
    validation_split=0.001,
    subset="training",
    seed=random_seed,
    image_size=(image_size, image_size),
    batch_size=batch_size)
print(train_ds0)
class_names = train_ds0.class_names
print(f"class list : {class_names}")

train_ds0 = train_ds0.prefetch(buffer_size=32)
val_ds0 = val_ds0.prefetch(buffer_size=32)

resnet = ResnetModel(num_classes=len(class_names), input_shape=(image_size, image_size, 3), resnet=model_input,
                     data_aug=Aug,weights=Weights,flatten=Faltten,dual=Dual)
model = resnet.Resnet_build()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss="categorical_crossentropy", metrics=['accuracy'])
#tf.keras.optimizers.Adam()
#tfa.optimizers.AdamW(learning_rate=learning_rate,weight_decay=weight_decay)
model.summary()

epochs = 25
floder = '1400file'
model_name = model_input._keras_api_names[0].split('.')[-1]
model_params = {
    'model_input' : model_name,
    'input_shape' : (image_size, image_size, 3),
    'Aug' : Aug,
    'Weights' : Weights,
    'Faltten' : Faltten,
    'Dual' : Dual,
    'weight_decay' : weight_decay,
    'learning_rate' : learning_rate
}
floder_model = os.path.join(floder,model_name)
if not os.path.exists(floder_model):
    os.mkdir(floder_model)

save_floder = os.path.join(floder_model,str(len(os.listdir(floder_model))))
if not os.path.exists(save_floder):
    os.mkdir(save_floder)

with open(os.path.join(save_floder,'params.json'),'w') as f:
    json.dump(model_params,f)
    f.close()

copy_list = ['main.py','model_create.py']
for src in copy_list:
    shutil.copy(src,save_floder)



try:
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint((os.path.join(save_floder,f"best_ResNet_model.h5")),monitor='val_accuracy',mod='max',save_best_only=True),
        # tf.keras.callbacks.ModelCheckpoint((os.path.join(save_floder,"Recall.h5")),monitor='val_recall',mod='max',save_best_only=True),
        # tf.keras.callbacks.ModelCheckpoint((os.path.join(save_floder,"Precision.h5")),monitor='val_precision',mod='max',save_best_only=True)
    ]
    history = model.fit(
        train_ds0, epochs=epochs,callbacks=callbacks,validation_data=val_ds0,
    )
    np.save((os.path.join(save_floder,f'history.npy')),history.history)
    model.evaluate(pre_ds)
    model.save((os.path.join(save_floder,f'model.h5')))
except:
    if("best.h5" not in os.listdir(save_floder)):
        shutil.rmtree(save_floder,ignore_errors=True)
