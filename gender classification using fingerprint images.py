#!/usr/bin/env python
# coding: utf-8

# In[36]:


pip install opencv-python


# In[37]:


pip install --upgrade keras


# In[38]:


pip install tensorflow


# In[39]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os
import cv2

import matplotlib.pyplot as plt


# In[40]:


def extract_label(img_path,train = True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')

    if train:
      gender = etc[0]
    else:
      gender = etc[0]

    gender = 0 if gender == 'M' else 1
    return np.array([gender], dtype=np.uint16)


# In[41]:


img_size = 96


def loading_data(path,boolean):
    data = []
    for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img),boolean)
            
            data.append([label[0], img_resize ])
            if len(data) % 1000 == 0:
                print(len(data))
    return data


# In[60]:


Real_path = r"C:\Users\SAI\Downloads\archive (3)\SOCOFing\Real"
Easy_path = r"C:\Users\SAI\Downloads\archive (3)\SOCOFing\Altered\Altered-Easy"
Medium_path = r"C:\Users\SAI\Downloads\archive (3)\SOCOFing\Altered\Altered-Medium"
Hard_path = r"C:\Users\SAI\Downloads\archive (3)\SOCOFing\Altered\Altered-Hard"
Easy_data = loading_data(Hard_path,True)


# In[61]:


img, labels = [], []
for label, feature in Easy_data:
    labels.append(label)
    img.append(feature)


# In[62]:


train_data = np.array(img).reshape(-1, img_size, img_size, 1)
train_data = train_data / 255.0


# In[63]:


from keras.utils import to_categorical
train_labels = to_categorical(labels, num_classes = 2)


# In[64]:


labels = np.array(labels)


# In[65]:


plt.imshow(train_data[5000])


# In[66]:


import tensorflow.keras as keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, #BatchNormalization, Dropout
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = Sequential([
Conv2D(32, 3, padding='same', activation='relu',kernel_initializer='he_uniform', input_shape = [96, 96, 1]),
MaxPooling2D(2),
Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', activation='relu'),
MaxPooling2D(2),
Flatten(),
Dense(128, kernel_initializer='he_uniform',activation = 'relu'),
Dense(1, activation = 'sigmoid'),
])
model.summary()


model.compile(optimizer = optimizers.Adam(1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# In[67]:


type(labels)


# In[68]:


# Counting the number of males and females
num_males = sum(1 for label in labels if label == 0)
num_females = sum(1 for label in labels if label == 1)

print("Number of males:", num_males)
print("Number of females:", num_females)


# In[69]:


history = model.fit(train_data, labels, batch_size = 128, epochs = 50, 
          validation_split = 0.2, callbacks = [early_stopping_cb], verbose = 1)


# In[70]:


test_data = loading_data(Real_path,False)

x_test,y_test= [], []
for label, feature in test_data:
    y_test.append(label)
    x_test.append(feature)


# In[71]:


x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)
type(y_test)


# In[72]:


model.evaluate(x_test,y_test)


# In[73]:


model.save('GenderFP.h5')


# In[74]:


def preprocess_image(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    img_resize = cv2.resize(img_array, (img_size, img_size))  # Resize to the required dimensions
    img_resize = img_resize.reshape(-1, img_size, img_size, 1)  # Reshape for model input
    img_resize = img_resize / 255.0  # Normalize the image
    return img_resize

# Load the model
model = tf.keras.models.load_model('GenderFP.h5')

# Path to the image you want to test
test_image_path = r'C:\Users\SAI\Desktop\p2.jpg'

# Preprocess the image
test_image = preprocess_image(test_image_path)

# Make predictions
prediction = model.predict(test_image)

# Print the prediction
print(prediction)

if prediction >= 0.5:
    print("The image is predicted to be female.")
else:
    print("The image is predicted to be male.")


# In[ ]:




