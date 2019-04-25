# exec(open('model_trainer.py').read())

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping


data = np.load("chess_games/train_dataset.npz")
samples = data['arr_0']
labels = data['arr_1']


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

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(samples, labels, batch_size=256, epochs=3, shuffle=True, callbacks=[early_stopping])

model.save("trained_models/model_chess_100.h5")

