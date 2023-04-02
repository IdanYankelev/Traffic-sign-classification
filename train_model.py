import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data = []
labels = []
num_of_classes = 43
cur_path = os.getcwd()

# Retrieving the images and their labels
for i in range(num_of_classes):
    path = os.path.join(cur_path, 'train', str(i))
    images_files = os.listdir(path)

    for image_file in images_files:
        try:
            image = Image.open(path + '\\' + image_file)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Converting the labels into one hot encoding and setting our X,y.
X_train, X_val, y_train, y_val = train_test_split(data, to_categorical(labels, 43), test_size=0.2, random_state=42)

# Building the model
model = Sequential()

# First Convolution Block
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second Convolution Block
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Dense Layers
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

# Define early stopping and learning rate scheduling callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

# Train the model with callbacks
history = model.fit(datagen.flow(X_train, y_train, batch_size=512), validation_data=(X_val, y_val), epochs=100
                    , verbose=2, callbacks=[es, lr_scheduler])
model.save("my_model.h5")

# Plot graphs for accuracy and loss
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
