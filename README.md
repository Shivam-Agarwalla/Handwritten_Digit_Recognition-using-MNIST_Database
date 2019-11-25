# Handwritten_Digit_Recognition-using-MNIST_Database

The MNIST dataset is a set of 70,000 human labeled 28 x 28 greyscale images of individual handwritten digits. It is a subset of a larger dataset available from NIST - The National Institute of Standards and Technology.

Introduction

Deep Learning employs subsequent layers of computation that break down the feature space into what are known as higher order feature representations. Here we will be training a deep learning model known as a Multi Layer Perceptron to recognize the numbers in handwritten digits. For this problem, a classifier will need to be able to learn lines, edges, corners, and a combinations thereof in order to distinguish the numbers in the images.At each layer, the learner builds a more detailed depiction of the training data until the digits are readily distinguishable by a Softmax output layer at the end.

Procedure To Run the Program

Open the python console and enter the following commands:

import dataset
training_data, validation_data, test_data = dataset.getdata()
import network
import imagetogreyscale
net=network.Network([784,30,10])
net.SGD(training_data,30,10,test_data=test_data)
test=imagetogreyscale.data("<Image File Name>")
net.predict(test)

