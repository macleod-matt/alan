
# Meet Alan
![Cover Photo](media/alanImage.jpg)


## Overview

This project involves the creation of a Convolutional Neural Network (CNN) trained on a custom dataset of playing cards. The goal of the project is to accurately classify playing cards, detect them within an image using bounding boxes, and maintain a running count, particularly useful for games like Blackjack.

The model was trained using the YOLOv8 (You Only Look Once) architecture, which is well-suited for real-time object detection tasks. By leveraging the power of YOLOv8, this project is capable of identifying and counting cards in real-time. And keeping a running count.


- **Classification**: Identifying the specific playing card present in an live video image. With the identified card, and percent confidence score.
- **Bounding Box Detection**: Drawing bounding boxes around detected cards within an image or video feed.
- **Running Count for Blackjack**: Keeping a real-time count of cards, a technique often used in Blackjack to improve game strategy.

This comprehensive setup allows for an automated, intelligent card detection system, suitable for gaming, statistical analysis, or other applications where card detection and counting are needed.

# Video Demo 
![Demo GIF](media/demo.gif)

#Files in This Repository

### 1. `alan.py`
This Module  contains utilities and helper functions used across the project. It includes various utility functions that support image processing, data manipulation, and other common tasks required by the other scripts.

### 2. `cardClassifier.py`
This Module  is responsible for classifying cards based on the trained model. It processes images, extracts relevant features, and predicts the class (i.e., the specific card) using the machine learning model.

### 3. `cardModelTrainer.py`
This Module  is used to train the machine learning model on a dataset of card images. It includes data preprocessing, model architecture setup, training loop, and model evaluation.

### 4. `countCards.py`
The purpose of This Module  is to count the number of cards present in a given image. It uses image processing techniques to detect and count the cards accurately.

### 5. `webcamCardClassifier.py`
This Module  extends the functionality of `cardClassifier.py` by allowing real-time classification of cards using a webcam. It captures video frames from the webcam, processes them, and classifies the cards in real-time.

### 6. `requirements.txt`
This file lists all the dependencies required to run the scripts in this project. Ensure that you install these packages using `pip` to avoid any compatibility issues.

## Quick Start Guide

```bash
1. git clone https://github.com/macleod-matt/alan.git

2. pip install -r requirements.txt

3. python alan.py

```

Next Updates: 
1. Multiple deck live running count count and true count
2. Incorproation with basic strategy allowing you insights when to hit/stay/raise bet 
3. Text to Speech engine advising hit, stay, raise bet based off the true count
