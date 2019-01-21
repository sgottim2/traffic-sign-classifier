# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data.jpg "Visualization"
[image2]: ./examples/dataexp.jpg "Data Distribution"
[image3]: ./examples/preprocess.jpg "Preprocessed Image"
[image4]: ./examples/LeNet_Original_Image.jpg "LeNet Architecture"

[image5]: ./test1/bumpy.jpg "Traffic Sign 1"
[image6]: ./test1/pedes.jpg "Traffic Sign 2"
[image7]: ./test1/roadwork.jpg "Traffic Sign 3"
[image8]: ./test1/speed70.jpg "Traffic Sign 4"
[image9]: ./test1/wildcrossing.jpg "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this is in the third cell of the ipython notebook. I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a visualization of single image from each class. It pulls in a random images and labels from each class with the correct names in reference with the csv file to their respective id's.

![alt text][image1]

Below image bar chart shows the data distribution. We can notice that the data distribution is not uniform.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the next step, I converted the images to grayscale. I followed the technical paper by LeCunn to preprocess the data. I believe 3 channels are not necessary for this problem. After that, I applied histogram equilization to the dataset. Here is an example of the final preprocessed image.

![alt text][image3]

After that, I normalized the data to improve the training performance. The image data should be normalized so that the data has mean zero and equal variance. Next, I augmented the dataset by applying random rotations. 

The difference between the original data set and the augmented data set is the following. The original training data has 34799 images and the augmented dataset has 46714 images in it. This helps the model much more effective.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used LeNet model architecture to implement this problem. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,    outputs  5x5x6 				|
| Flatten   	      	| outputs  400                      			|
| Fully Connected      	| outputs  120                      			|
| RELU					|												|
| dropout				|												|
| Fully Connected      	| outputs  84                       			|
| RELU					|												|
| dropout				|												|
| Fully Connected      	| outputs  10                       			|
|						|												|
|						|												|
 
 
 Here is the visualization of the LeNet Architecture:
 
 ![alt text][image4]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Before training the model, I shuffled the training dataset. To train the model, I used LeNet model. I uesd an adam optimizer with a learning rate of 0.0005 for 40 epochs. The batch size is 128. I plyed around with various values and I found these to be optimal parameters. With these parameters I was able to get 95% accuracy on the validation dataset. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 95%
* test set accuracy of 93%

First, I used the model which was taught on Udacity. Then I modifed the LeNet model by adding a dropout layers to it. This improved the model accuracy. As mentioned in the techncal paper, this architecture is relevant to the classification problem. There are many other advanced model available. I chose this model because it was simple, faster and acheived the desired results.
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| Pedestrians  			| Pedestrians									|
| Road work				| Beware of ice									|
| 70 km/h	      		| 70 km/h       				 				|
| Wild Crossing			| General caution      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Testing on new images cell of the Ipython notebook.

The following shows the first five softmax probabilities for each image.

Results for image 0

0.56042 Go straight or left
0.38165 Speed limit (70km/h)
0.05604 Traffic signals
0.00085 Speed limit (30km/h)
0.00080 General caution

Results for image 1

0.42042 Wild animals crossing
0.33802 Slippery road
0.15244 Double curve
0.08911 Dangerous curve to the left
0.00000 Dangerous curve to the right

Results for image 2

1.00000 Bumpy road
0.00000 Bicycles crossing
0.00000 Children crossing
0.00000 Wild animals crossing
0.00000 Go straight or right

Results for image 3

0.65763 General caution
0.23569 Traffic signals
0.10572 Pedestrians
0.00082 Road narrows on the right
0.00008 Right-of-way at the next intersection

Results for image 4

0.80497 Beware of ice/snow
0.19381 Right-of-way at the next intersection
0.00110 Road work
0.00007 Road narrows on the right
0.00004 Children crossing


