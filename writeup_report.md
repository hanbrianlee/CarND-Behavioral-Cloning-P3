# **Behavioral Cloning** 



Write-up: BRIAN 'HAN UL' LEE



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the final script to create and train the model
* clone.py containing the script that I used to test/develop the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.py containing the script that compiles images created by drive.py into an mp4 video file
* video.mp4 containing the video of one-lap around Track 1 by my model (model.py)
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first try the bare-bone flatten layer as the tutorials introduced with the udacity provided datasets and then try more sophisticated models. 
The first shot was actually not too bad. The car did stay within the road except that it was right at the limit of the road (on the right side). However, as it went to the sections where the red dashed sidelines are, the car increasingly drove to the right side limit of the road and eventually when sharper corners, came, it went off the road.
Also, with this bare-bone model, adding the left and right camera data seemed to worsen the results, and the car was going off the road right from the get-go into the mountains and into the lake, etc. I believe this was due to the fact that the images were not cropped and thus adding the left and right camera images pounded on more environment noises (trees, lakes, etc.) than before.

I then implemented the NVIDIA deep neural networks published on the internet. Also, I recorded my own training data, which I purposely made it very short and I purposely drove it off the road to see if my network will drive the car off the road, and it did.
The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

For pre-processing, I tried to add flipped images as well but it exceeded my memory and my pyCharm failed with memory error. I considered using the amazon services but I wanted to see if my computer can handle this task.
I went onto implement the generator for my model, put batching scheme in place, and using fit_generator I was able to only use a max of 1GB on my computer and the model could be trained properly. I used verbose=1 option in fit_generator to be able to see the progress and how much time is left for training.

I then went on to record a more proper data, but my computer's spec was not great and the simulator lagged so bad that I could not drive a full lap without going off the road. I tried my best however and saved around a half-lap of data.
However, visualizing graph of train and validation results against epochs showed that while training error consistently decreased while validation error was bouncing up and down. So I just went and recorded more data.
I recorded around 3 laps of data to provide enough data to avoid overfitting, and it seemed to work fine without the need of a drop-out layer.

The final step was to run the simulator to see how well the car was driving around track one. 
The car actually drove very smoothly and I was impressed by my model. The car almost always stayed dead center and even when I take over and drive the car nearly off the road or slightly off the road, the car managed to find itself right back to the center in smooth fashion.


#### 2. Final Model Architecture

My final model (model.py line 49~81)closely resembles the NVIDIA self-driving car neural network published here (nvidia architecture)https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
The model includes RELU to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (model.py line 70).

There were additional implementations I was considering to make such as follows, but I found out that my model handled Track 1 so nicely that any further additions seemed like they won't improve it much. There's the good old engineering saying, if it ain't broken, don't fix it.
1. Adding drop-out layer: perhaps this could even make the autonomous drive even more smooth
2. Try more epochs: this could also help my model learn more precisely to follow my driving behaviour, but it should not be too high because the model can overfit
3. Flip augmentation: this track mostly runs in the counter-clockwise direction, but maybe on a different track that is mostly clock-wise, the flip augmented data will help my model generalize better and drive well on both clockwise and counter-clockwise tracks
4. Adding distorted images using keras data generator to ensure there are equally many soft-curve and hard-curve roads as there are straight-roads.

#### 3. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road without going astray from the road. 
I just used the basic feature of human (me), namely inconsistency. There was no way I was going to achieve the exact same cornering and center driving on around 3 laps consecutively. Yes, it is technically possible to achieve near-identical driving behaviour, but I just let my self go and drove casually as an average human would.
Hence, if you can see my training data, for the most part it does drive in the center of the road in straight roads and curved roads alike, sometimes I veer off to the limits of the roads and make mistakes and recover. These subtle mistakes helped my training datasets to capture various "views" of the same roads and therefore the "right" steering angle actions to be taken to bring it to the center.

I randomly shuffled the data set for each batch using sklearn.utils.shuffle (model.py line 58). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
For epochs, I stuck with 3 since my computer doesn't have a GPU and I could not wait too long between each training. But 3 seemed to work splendidly. 
I used an adam optimizer so that manually training the learning rate wasn't necessary (model.py line 84).