# Behavioral Cloning 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[crop]: ./examples/crop.png "Crop"
[hist1]: ./examples/hist1.png "Histogram before"
[hist2]: ./examples/hist2.png "Histogram after"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `Behavioural cloning.ipynb` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [Behavioural cloning.ipynb](./Behavioural%20cloning.ipynb) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have defined a multilayer convolutional neural network with fully connected layers at the head. I use ELUs as activation function as they give slightly better performance than ReLUs. 

I have considered that for the sake of simplicity, it is better to crop and standarize the images in the model itself, using Keras layers. This way, during the inference phase, we can feed the model directly with the images captured from the simulator.

#### 2. Attempts to reduce overfitting in the model

`Dropout` layers are used to prevent overfitting in all the layers but the first. 

Also, I split the data into a training and a validation set. This way I can monitor the learning curves and stop the training in case the model start to overfit. This can also be done automatically with a Keras callback. As I have lots of data samples, I decided to use 10% of the data for validation.

#### 3. Model parameter tuning

I use MSE as loss function, and an `Adam` optimizer with `learning rate = 0.0005` and setting `beta1` and `beta2` to their default values. After some experiments, I noticed that lowering the learning rate improved the accuracy of the driving.

I trained my model during 20 epochs with a batch size of 64. After that, I realized that I could keep training for longer, as both loss curves were still improving. Fortunately, Keras allows to extend the training without breaking the model. The final number of epochs was 30.

I also added a Keras callback to reduce the learning rate in case that a plateau was found (eventually, this wasn't necessary).

#### 4. Appropriate training data

I used both circuits in both senses to record driving data. I also recorded myself driving from the sides to the center of the road.

For details, see the python notebook and next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

From the beginning, I started with a similar architecture as the final one, but with ReLUs activations and less Convolutional and Dense layers. I kept adding complexity to the model while preventing overfitting by means of Dropout layers till I couldn't improve the validation losses and got a good performance at the simulator. I also experimented with `MaxPool` layers, but I opted for a full-convolutional architecture.

After each training I tested the model in the simulator. I found that there were some problematic points in the circuit, e.g. near the lake. So I decided to record more samples at these points, solving the problem. Also, using the second circuit helped a lot for the training. I imagine that it helped the model to generalize better.

Using the left and right cameras was a huge improvement also. At the beginning I used a large value for the correction. This made the car to oscillate a lot, but after reducing the correction angle, this problem was eliminated.

#### 2. Final Model Architecture


```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 160, 16)       1216      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 80, 32)        12832     
_________________________________________________________________
dropout_1 (Dropout)          (None, 17, 80, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 40, 64)         51264     
_________________________________________________________________
dropout_2 (Dropout)          (None, 9, 40, 64)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 20, 128)        204928    
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 20, 128)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 10, 256)        295168    
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 10, 256)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 3, 10, 128)        32896     
_________________________________________________________________
dropout_5 (Dropout)          (None, 3, 10, 128)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 3840)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               491648    
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_7 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 64)                4160      
_________________________________________________________________
dropout_8 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 32)                2080      
_________________________________________________________________
dropout_9 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,104,481
Trainable params: 1,104,481
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

I load diferent training sessions and concatenate them. `data2` is recorded on the mountain circuit, in both senses, and I find that it is better to reduce slightly the magnitude of the angle correction. The rest are loaded in the first circuit, in both senses too. 

Due to the nature of the problem there are lots of samples where no steering is performed, i.e. while driving in a straight line. I have found that removing these samples improves the performance of the model. 

![hist1]

![hist2]

In order to reduce the size of the samples and consequently the model, I have considered convenient to remove the top and the bottom parts of the images. This will remove useless information and help the model to generalize better. It also hides the hood of the car that could be used by the model to cheat and guess the camera that was used.

The cropping process is incorporated into the model, using a `Cropping2D` layer from Keras. After some experimentation, I decided to crop 75 pixels from the top and 25 pixels from the bottom.

![crop]

After the cropping, the images are standarized, scaling their values to the range [-0.5, 0.5]. For this, a Keras `Lambda` layer is used.
