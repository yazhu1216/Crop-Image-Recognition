import tensorflow as tf
model = tf.keras.models.load_model('./best_mix_model.h5')
image_size = 672
ta_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    '../predict',
                    label_mode="categorical",
                    validation_split=0.1,
                    subset="validation",
                    shuffle=False,
                    image_size=(image_size,image_size),
                    batch_size=7)

pre_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '../predict',
    label_mode="categorical",
    validation_split=0.1,
    subset="training",
    shuffle=False,
    image_size=(image_size,image_size),
    batch_size=7)

import numpy as np
classname = pre_ds.class_names
predict = np.array([])
labels = np.array([])
for x,y in pre_ds:
    predict = np.concatenate([predict,np.argmax(np.array(model.predict(x)),axis=1)])
    labels = np.concatenate([labels,np.argmax(y.numpy(),axis=1)])
for x,y in ta_ds:
    predict = np.concatenate([predict,np.argmax(np.array(model.predict(x)),axis=1)])
    labels = np.concatenate([labels,np.argmax(y.numpy(),axis=1)])
pref = (predict == labels)
print(len(pref))
print(pref.sum()/len(pref))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predict, labels)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(labels, predict,target_names=pre_ds.class_names))