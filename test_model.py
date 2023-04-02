import keras
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import random

# Load best model
model = keras.models.load_model('my_model.h5')

# Test accuracy on test dataset
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

# Predict labels for test dataset
pred = model.predict(X_test)
pred = np.array([np.argmax(EncodedPred) for EncodedPred in pred])

# Accuracy of the test data
ConfusionMatrixDisplay.from_predictions(labels, pred)
plt.show()
print(classification_report(labels, pred))

#  Display random data examples and the comparision of actual vs predicted label.
random_indices = random.sample(range(len(pred)), 25)
subplot_index = 0
for i in random_indices:
    plt.subplot(5, 5, subplot_index + 1)
    subplot_index += 1
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[i]
    actual = labels[i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color=col)
    plt.imshow(X_test[i])
plt.show()

#  Display subset of data examples which the model was wrong in their prediction.
wrong_indices = [i for i, v in enumerate(pred) if pred[i] != labels[i]]

subplot_index = 0
for i in wrong_indices[:25]:
    plt.subplot(5, 5, subplot_index + 1)
    subplot_index += 1
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[i]
    actual = labels[i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color=col)
    plt.imshow(X_test[i])
plt.show()
