#**Behavioral Cloning** 

Writeup Template
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/bc2_3epoch.png "Keras History - 3 epochs, successful, four rounds forward, additional images from difficult turns, all images flipped."
[image3]: ./images/bc2_8epoch.png "Keras History - 8 epochs, not always successful"
[image4]: ./images/PlayedByAndreasWithAdditionalTurns.png "Trained with images from Track 1, forward, back and additional images from difficult turns"
[image5]: ./images/T2a.png "Training with images from Track 2"
[image6]: ./images/T2b.png "Training with images from Track 2 - Augmented"
[image7]: ./images/Track1Forth4.png "Trained with images from Track 1, 4 rounds forward"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* images folder contains line charts from keras history while training
* video.mp4 is the generated video showing the agent's view for the reviewer
* video.py is the tool used to create the video.mp4
* carnd_term1_p3.webm is a video showing the screen capture while running the agent using model.h5 for two rounds in a row (full speed), to demonstrate the robustness of the solution (to be uploaded later because of size constraints)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model closely follows the NVIDIA's paper with the following additions:

* I used Lambda layer to normalize input images plus to avoid saturation and make gradients work better.
* I used Lambda layer to crop the top of the images (to eliminate unwanted pixels including trees etc) and the bottom of the image (the nose of the vehice)
* I've added an additional dropout layer to avoid overfitting after the convolution layers.
* I've also included RELU for activation function for all convolutional layers to introduce non-linearity.

####2. Attempts to reduce overfitting in the model

I've added an additional dropout layer to avoid overfitting after the convolution layers. (line 273) 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 80-86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 292).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Initially i used center lane driving, with addition of images for difficult turns all in a single folder.
But then i realized the importance of having the flexibility to incrementally add training images for debugging purposes. The result was the function read_csv (line 32).
The main idea is to build an array of lines from the csv file and enrich the columns with tags that will be used later as instructions while training


For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the NVIDIA paper model because it seemed a good candidate, built for the purpose of end to end self driving test.
Once the selection was made, i had to adjust the images to achieve the best result as well as to introduce non-linearity and avoid overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (line 88). 
In general, i was surprised to see that i was achieving promising results with small epoch numbers (3-8). But usually the validation error was smaller than the training error (indication of underfitting).

The car was falling off the track... I used several different data directories trying to figure out the best solution.

Some examples:

Trained with images from Track 1, 4 rounds forward
![alt text][image7]

Trained with images from Track 1, forward, back and additional images from difficult turns
![alt text][image4]

To combat the cases of overfitting, I modified the model so that with a dropout right after the convolutional layers, as described earlier.

This was one of the successful scenarios:
Keras History - 3 epochs, successful, four rounds forward, additional images from difficult turns, all images flipped.
![alt text][image2]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Interestingly enough the same actions with 8 epochs training lead to a result having 50% chance of failure:
Keras History - 8 epochs, not always successful
![alt text][image3]
The undefitting is obvious

Following the success in track one, i repeatedly tried to train a more generalized model for Track 2, with not good results.

I modified the code in the following ways:

* Randomly choose right, left or center images.
* For left image, steering angle is adjusted by +0.2 (This is used for Track 1 too)
* For right image, steering angle is adjusted by -0.2 (This is used for Track 1 too)
* Opt to flip image left/right
* Randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift) : I never felt i gained something in this way, the code is commented out
* Randomly translate image vertically : I never felt i gained something in this way, the code is commented out
* Randomly added shadows
* Randomly altering image brightness (lighter or darker)

Training with images from Track 2
![alt text][image5]

Training with images from Track 2 - Augmented
![alt text][image6]

Indeed the epochs value was low, also i used only three rounds of forward drive plus flipping all images, plus augmenting as described.
As i stated, no success with Track 2.

I noticed that some people were able to train using images from Track 1 and test on the older Track 2 having success. I believe that this is
not possible with the new track: One, needs to gather data from Track 2 as well, also train with enough data and large number of epochs (there the power of augmentation will be proved)


####2. Final Model Architecture

The final model architecture (model.py lines 262-284) implements the NVIDIA paper

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

Most of this part is already described above. In summary, i used center lane driving and repeating recording for difficult turns, but i did not do recovering.
Then i combined different recording scenarios to train. I augmented the data using flipping and shuffled the csv lines (essentially leading to image shuffling).
I used images from all three cameras. For the successful scenarion presented here, i used i general 26250 images with number of epochs = 3.