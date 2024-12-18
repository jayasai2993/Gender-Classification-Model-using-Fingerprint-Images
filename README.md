# Gender-Classification-Model-using-FingerprintImages-and-Deep-Learning <br>[https://www.linkedin.com/posts/jayasai2993_python-tensorflow-keras-activity-7225736549051592704-uGFc?utm_source=share&utm_medium=member_desktop]
Gender Classification Using Fingerprint Images with Deep Learning <br><br>
This project demonstrates a gender classification system based on fingerprint images using a Convolutional Neural Network (CNN). The model is trained using TensorFlow/Keras and deployed as a web application using Flask. Users can upload fingerprint images to classify them as "Male" or "Female" with real-time predictions. <br><br>

Features <br><br>
Deep Learning Model: <br>

Built with TensorFlow and Keras. <br>
Utilizes a CNN architecture with two convolutional layers, max-pooling, and dense layers. <br>
Designed for binary classification with a sigmoid activation function in the output layer. <br>
Web Interface: <br>

Developed using Flask. <br>
Allows users to upload fingerprint images for gender classification. <br>
Displays the uploaded image and prediction results with visual feedback. <br>
Dataset: <br>

Based on the SOCOFing Dataset. <br>
Includes real and altered fingerprint images for robust training and evaluation. <br>
Model Architecture <br>
The CNN model consists of: <br>

Convolutional Layers: Extract spatial features from input images. <br>
Max-Pooling Layers: Reduce spatial dimensions and computational load. <br>
Dense Layers: Perform high-level feature learning and binary classification. <br>
Sigmoid Activation: Provides probabilistic output for gender prediction. <br>
Technologies Used <br>
Programming Languages: Python <br>
Libraries: <br><br>
TensorFlow/Keras: For building and training the deep learning model. <br>
OpenCV: For image preprocessing (grayscale conversion, resizing, normalization). <br>
Flask: For creating the web application. <br>
Front-End: HTML, CSS, and Jinja2 templates for dynamic rendering. <br>
How It Works <br>
Training: <br><br>
Fingerprint images are preprocessed (grayscale, resized to 96x96 pixels, normalized). <br>
The CNN model is trained on the dataset with labels for gender (Male: 0, Female: 1). <br><br>
Deployment: <br>

The trained model is saved as GenderFP.h5. <br>
Flask serves a web interface where users can upload images. <br>
The uploaded image is preprocessed, and the model makes predictions. <br><br>
Prediction: <br>

A probability score is generated by the model. <br>
Threshold: <br>
>= 0.5: Predicted as "Female". <br>
< 0.5: Predicted as "Male"..... <br>
