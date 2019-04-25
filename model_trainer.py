# exec(open('chess_model_trainer.py').read())

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping


data = np.load("train_dataset_2.npz")
samples = data['arr_0']
labels = data['arr_1']

# samples_split = np.array_split(samples_raw, 10)
# labels_split = np.array_split(labels_raw, 10)
# samples = tf.keras.utils.normalize(samples_split[0], axis=1)
# labels = labels_split[0]

# count = 0
# for sample in samples_split:
    # count += 1
    # if count == 1:
        # continue
    # print("normalizing partition #" + str(count))
    # sample = tf.keras.utils.normalize(sample, axis=1)
    # samples = np.concatenate((samples, sample), axis=0)

# sample_1 = samples_split[0]
# sample_2 = samples_split[1]
# sample_3 = samples_split[2]
# sample_4 = samples_split[3]
# sample_5 = samples_split[4]
# sample_6 = samples_split[5]
# sample_7 = samples_split[6]
# sample_8 = samples_split[7]
# sample_9 = samples_split[8]
# sample_10 = samples_split[9]

# sample_1 = tf.keras.utils.normalize(sample_1, axis=1)
# sample_2 = tf.keras.utils.normalize(sample_2, axis=1)
# sample_3 = tf.keras.utils.normalize(sample_3, axis=1)
# sample_4 = tf.keras.utils.normalize(sample_4, axis=1)
# sample_5 = tf.keras.utils.normalize(sample_5, axis=1)
# sample_6 = tf.keras.utils.normalize(sample_6, axis=1)
# sample_7 = tf.keras.utils.normalize(sample_7, axis=1)
# sample_8 = tf.keras.utils.normalize(sample_8, axis=1)
# sample_9 = tf.keras.utils.normalize(sample_9, axis=1)
# sample_10 = tf.keras.utils.normalize(sample_10, axis=1)

# samples = np.concatenate((sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8, sample_9, sample_10), axis=0)



model = Sequential()
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.55))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
# Adam(lr=.0001)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.compile(optimizer='rmsprop',
              # loss='binary_crossentropy',
              # metrics=['accuracy'])


# from keras.optimizers import SGD
# opt = SGD(lr=0.01)
# model.compile(loss = "categorical_crossentropy", optimizer = opt)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(samples, labels, batch_size=256, epochs=1000, shuffle=True, callbacks=[early_stopping])

model.save("model_chess_final_4.h5")

