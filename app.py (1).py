#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask , render_template , request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
app=Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\SAI\GenderFP.h5')

def preprocess_image(image_path):
    img_size=96
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    img_resize = cv2.resize(img_array, (img_size, img_size))  # Resize to the required dimensions
    img_resize = img_resize.reshape(-1, img_size, img_size, 1)  # Reshape for model input
    img_resize = img_resize / 255.0  # Normalize the image
    prediction = model.predict(img_resize)
    return prediction

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path1 = imagefile.filename
    save_path='./static/'+imagefile.filename
    print(save_path)
    imagefile.save(save_path)
    prediction = preprocess_image(save_path)
    if prediction >= 0.5:
        c="female"
        d="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5s87ekSAHkGx0aa-banQzih9cxROEE7wDKwaGlnhNBweaIAXz397Fa1KhrZnIXcY1-E0&usqp=CAU"
    else:
        c="male"
        d="https://cdn1.iconfinder.com/data/icons/user-pictures/101/malecostume-512.png"

    return render_template('output.html',prediction=c,path=save_path,g=d)

if __name__ == '__main__':
    app.run(port=5001)


# In[ ]:




