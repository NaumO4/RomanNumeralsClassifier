# RomanNumeralsClassifier
Test task for hackaton INT20H 2019

The program recognizes handwritten Roman numerals from 1 to 8,
using TensorFlow.

![alt text](pictures/examples/1_4.jpeg) ![alt text](pictures/examples/2_4.jpeg) ![alt text](pictures/examples/3_3.jpeg) ![alt text](pictures/examples/4_4.jpeg)\
![alt text](pictures/examples/5_5.jpeg) ![alt text](pictures/examples/6_5.jpeg) ![alt text](pictures/examples/7_5.jpeg) ![alt text](pictures/examples/8_5.jpeg)

To run the program you need to install dependencies  `$pip install -r requirements.txt`. Train and test the network in the jupyter notebook run.ipynb

The network classifies numbers on a test data with an accuracy of 0.9975

**Preparation of data**
Images was augmented using flipping, rotation and scaling