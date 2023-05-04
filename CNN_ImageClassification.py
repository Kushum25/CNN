
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:23:21 2023

@author: Kushum
"""

# Step 1: Loading Libraries
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

#Step 2: Loading and reading the image data
#Step 2.1: Define image directory for 2 diifernt folder of images 
cracked_dir = "D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Assignment_7\\WD\\Cracked"
uncracked_dir = "D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Assignment_7\\WD\\Uncracked"

#Step 2.1: Load and process image 
def load_and_preprocess_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.resize(img, (256, 256))  # Resize the image to 256x256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        
        # Rotate the image by 90 degrees counterclockwise
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_img = rotated_img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        
        # Append the preprocessed images to the list
        images.append(img)
        images.append(rotated_img)
        
    return np.array(images)
 
cracked_images = load_and_preprocess_images(cracked_dir)
uncracked_images = load_and_preprocess_images(uncracked_dir)


#Step 3: Splitting the data into training and testing sets
X = np.concatenate((cracked_images, uncracked_images), axis=0)
y = np.concatenate((np.ones(cracked_images.shape[0]), np.zeros(uncracked_images.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1234)

#Step 4: Building the CNN model
model = Sequential()

# Convo2D apply convolution operations with the specified number of filters and kernel size
# First convolutional layer with 16 filters, a 3x3 kernel, and a ReLU activation function
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)))     #input shape = img_width, img_height
# Add a max pooling layer with a 2x2 pool size
model.add(MaxPooling2D((2,2)))

# Second convolutional layer with 16 filters and a 3x3 kernel
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Third convolutional layer with 32 filters and a 3x3 kernel
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Fourth convolutional layer with 32 filters and a 3x3 kernel
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

# Flatten layer: to convert output of the convolutional layers to a single vector
model.add(Flatten())

# 2 Dense layers:fully connected layer with 32 neurons and a ReLU and sigmoid activation function
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#Step 5:Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Step 6: Fitting the model
history = model.fit(X_train, y_train, epochs=10, batch_size=16)


# Step 7: Evaluating the model on the testing data
#Step 7.1: Calculating loss function and accuracy of model in test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss on testing data:", loss)
print("Accuracy on testing data:", accuracy)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

# Step 7.2: ROC curve for model evaluation
#Generate predictions for the test data
y_pred = model.predict(X_test)

# Generate the ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.show()


# Step 8: Making predictions on a single image
img_path = "D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Assignment_7\\WD\\Test\\test_image.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0

img = np.expand_dims(img, axis=0)  # Add batch dimension

prediction = model.predict(img)
if prediction > 0.5:
    print("The image is cracked.")
else:
    print("The image is uncracked.")































