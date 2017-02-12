#**Behavioral Cloning** 

##Writeup

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used the NVidia model which has:
* 1 Lambda layer to normalize the data
* 3 Convolutional layers with 5x5 kernel and 2x2 stride
* 2 Convolutional layers with 3x3 kernel and no stride
* 5 Fully connected layers leading to the final output

I used relu for my activations.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers after each fully connected layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track counter-clockwise, driving the track focusing on smooth cornering, and specialized recordings of difficult areas (like the bridge)

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach



My first step was to use a convolution neural network model similar to the NVidia model. I thought this model might be appropriate because it was used for a similar task and it proved more effective than the comma.ai model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I included dropout from the beginning so it didn't really overfit until I started adding extra, but small datasets.

One of the problems I encountered was underfitting because I had a lot of dropout in my model to begin with, I reduced the percentage of dropout in the model and that seemed to help.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

####3. Creation of the Training Set & Training Process

The simulator proved to be difficult at times to control so I saved different runs in different locations.  This allowed me to record small sections such as a lap of the track or recovering from the sides and be able to easily remove data if I accidentally overcorrected or drove off the track.  This also allowed me to mix and match datasets so that I could train first on the Udacity provided data save and test the model, and then train again on problem areas like corners.  

I also kept a log of the most successful models (details about the model trained, which datasets were used and for how many epochs)

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
