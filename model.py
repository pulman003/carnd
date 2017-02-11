import pickle
import tensorflow as tf
import numpy as np
import os
import csv
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import load_model

tf.python.control_flow_ops = tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data', '', 'Data path')
flags.DEFINE_string('model', '', 'Model path')
flags.DEFINE_integer('epochs', 5, 'Number of epochs')
flags.DEFINE_integer('batch_size', 256, 'Batch size')

samples = []
with open(FLAGS.data+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = FLAGS.data+'IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            X_train = np.array(images)
            X_train = X_train[:,80:,:,:]
            y_train = np.array(angles)
            #print (X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_model(time_len=1):
  ch, row, col = 3, 80, 320  # camera format (scaled by 2)

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(ELU())
  model.add(Dense(100))
  model.add(Dropout(0.2))
  model.add(ELU())
  model.add(Dense(50))
  model.add(Dropout(0.2))
  model.add(ELU())
  model.add(Dense(10))
  model.add(Dropout(0.2))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")
  return model

#img = get_data('driving_log.csv', FLAGS.data)
#for image in img:
#    print(image)
train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
#print(next(train_generator))
validation_generator = generator(validation_samples, batch_size = FLAGS.batch_size)

model_path = FLAGS.model
if model_path:
    model = load_model(model_path)
    print("Loaded")
else:
    model = get_model()
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_epoch=FLAGS.epochs, validation_data=validation_generator, nb_val_samples = len(validation_samples))

model.save('model.h5')
