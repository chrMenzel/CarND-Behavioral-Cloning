# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Normal.jpg "Normal Image"
[image2]: ./examples/Flipped.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 a video of two rounds of autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model (model.py lines 94-120) is nearly identical to the Nvidia convolution neural network for self driving cars. I made only two changes:

- The data is normalized in the model using a Keras lambda layer (code line 96). 
- After normalization, the images are cropped (code line 97).
- Between every dense layer I inserted a dropout layer (code lines 113, 115 and 117).


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 113, 115 and 117). 

The model contains data augmentation in order to reduce overfitting (model.py lines 77-91).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 151). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 150).

#### 4. Appropriate training data

As training data I chose the sample data provided by Udacity. Before that I tried very often to create own data by driving the car in the simulator but I always failed to achieve a single round without troubles. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to go simple and adding complexity as needed.

My first step was to use a convolution neural network model similar to the LeNet network. I thought this model might be appropriate because it was known that it works well for traffic sign recognition. But traffic sign recognition is a classification problem, the problem here (training a model to predict the steering angles of a car to drive itself in the simulator) is a regression problem. So my next approach was the convolutional network from NVIDIA which is known for working well for self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added 3 dropout layers between the 4 dense layers and a cropping layer to get rid of the non-relevant parts of the image (trees, landscape etc.)

Additionally I added more training data by using the left and right camera images additionally to the center camera images and flipping the images horizontally. Then I also added a correction of the measured steering angle of 0.2.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in curves. To improve the driving behavior in these cases, it turned out that the correction of the steering angle (0.2 in my first attempt) was not enough and should not be used if the steering measurement was exactly or very near around zero.  So I corrected the steering angle only if the measured angle was less than -0.05 or greater than 0.05 and increased the correction from 0.22 over 0.25 to 0.3.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-120) consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)      		|     Output shape       					| 
|:---------------------:|:---------------------------------------------:| 
| lambda_1 (Lambda)         		| (None, 160, 320, 3)   							| 
| cropping2d_1 (Cropping2D)     	| (None, 65, 320, 3) 	|
| conv2d_1 (Conv2D) 				|	(None, 31, 158, 24)											|
| conv2d_2 (Conv2D) 	      	| (None, 14, 77, 36)				|
| conv2d_3 (Conv2D)     | (None, 5, 37, 48)      									|
| conv2d_4 (Conv2D) 					|	(None, 3, 35, 64)											|
| conv2d_5 (Conv2D) 	      	| (None, 1, 33, 64) 				|
| flatten_1 (Flatten)			|	(None, 2112)											|
| dense_1 (Dense)  | (None, 100)      									|
| dropout_1 (Dropout)		| (None, 100)        									|
| dense_2 (Dense)  | (None, 50)      									|
| dropout_2 (Dropout)		| (None, 50)        									|
| dense_3 (Dense)  | (None, 10)      									|
| dropout_3 (Dropout)		| (None, 10)        									|
| dense_1 (Dense)  | (None, 1)      									|


#### 3. Creation of the Training Set & Training Process

I had no success to capture good driving behavior, neither with mouse nor with the arrow keys. So I only used the sample data provided by Udacity.
To augment the data set, I also flipped images and angles (see above).

Here is an example of a normal image and a flipped image:
![Normal][image1]
![Flipped][image2]

After the collection process, I had  I had 48.216 images and steering angles. I finally shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trying out. I used an adam optimizer so that manually training the learning rate wasn't necessary.
