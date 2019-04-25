# exec(open('model_load.py').read())

import keras
from keras.models import load_model
import numpy as np



test_data = np.load("test_dataset.npz")
test_samples = test_data['arr_0']
test_labels = test_data['arr_1']

model = load_model('model_chess_final_2.h5')

# predictions = model.predict(test_samples, batch_size=2)
predictions = model.predict_classes(test_samples, batch_size=2)

count = 0
score = 0
for i in predictions:
    # print(i)
    if i==test_labels[count]:
        score += 1
    count+=1
    # if count > 4000:
        # break
print("Score is: " + str(score) + " out of " + str(count))
percent = (score/count)*100
print("Percent correct: " + str(percent))