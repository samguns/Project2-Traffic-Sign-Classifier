# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/train_orig_barchart.png "Original Training Set"
[image2]: ./images/feature_map.png "Conv1 feature map"
[image3]: ./images/feature_map_l2.png "Conv2 feature map"
[image4]: ./images/download_signs.png "Downloaded Traffic Signs"
[image5]: ./images/all_classes_example.png "All Classes"
[image6]: ./images/80kmh.png "80"
[image7]: ./images/roadwork.png "Road Work"
[image8]: ./images/stop.png "Road Work"
[image9]: ./images/yield.png "Road Work"
[image10]: ./images/roadwork.png "Road Work"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy & csv library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43
![image5]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Most classes do not have many samples but some have more than 1500.

![image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data by dividing 255 to reduce large variances. It makes a guaranteed stable convergence of weight and biases. This pre-processed training set gave me a faster and acceptable result. However, even my training model reached an accuracy about 94% on validation set, but performed poorly on test sets. I tried increasing epochs as well as tuning hyper-parameters. Hardly any noticeable improvement I could get.

After I implemented the feature map visualization, it dawned on me that it might worth trying to convert image into grayscale. For it filters out color information which does little help but occupies more memory space.

I also tried to augment the training set, because I think an evenly distributed classes would help to generalize the model. The augmented training set didn't contribute a more accurate results, so I didn't pursue that approach. Maybe I should try harder later.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 Grayscale image                               |
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x18     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x18                 |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x36 |
| RELU        |                      |
| Max pooling              | 2x2 stride,  outputs 5x5x36                 |						  
| Fully connected        | 900 units                                            |
| RELU   |      |
| Fully connected     | 300 units    |
| Dropout      | 25% keep probability for training set  |
| Softmax                | Output the top probable prediction of the 43 classes   |                                   



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epoch of 5 with 128 samples in each batch. The optimizer is Adam and learning rate is set to be 0.001. The keep probability for Softmax layer input is 25% for training, 100% for evaluation.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 96%
* test set accuracy of 94%

In the kind of trial and error iterative process, I first chose LeNet5 learned in the course as my starting point. It’s easy to implement, and most of all, the image is 32x32x3, which is perfectly fit into the code I’ve done, with just some small changes.

The results were hardly acceptable, for it suffered from overfitting easily, the validation accuracy was low, let alone performance on test sets.

So I started increasing filter sizes in Convolutional layers. It improved a bit but not much. Validation accuracy was barely pass the minimum requirement and still, the more epochs it ran, the more it overfit. I couldn’t observe any significant improvements.

To mitigate the overfitting problem, I considered adding a dropout layer before the last fully connected layer. I first tried 80% keep probability, it worked. So I continued lower the value until it reached 25%. By then I got a satisfied performance on both validation and test sets. Finally I settled with the grayscale input LeNet5 model stated above.

Because this course provided an access to GPU training, I tried an [ResNet50] model on this task also. Aside from it consumed much more time on training, it didn't perform better than simple models such as LeNet5. So I didn't continue in pursuing that approach.
If a well known architecture was chosen:


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image4]

The 80kmh sign might be difficult to classify since it's a bit darker, but as the feature map and predictions shown the model did well on it.

At this stage, I think I didn't choose the Road Work and Stop wisely. They seems to be easy for classification. But nonetheless, it made a mistake on predicting Stop sign.

The Yield sign could be a pitfall because it has a little line inside the inner triangle.

The Traffic sign is not clear enough and the details in the middle might easily confused the model to treat it as many other signs, such as General caution or Double curve.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| **Stop Sign**              | 30km/h |
| 80km/h                 | 80km/h |
| Yield                  | Yield  |
| Traffic signals        | Traffic signals  |
| Road work              | Road work  |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94% is a bit lower. The precision for Stop sign is unreliable, for I ran the classification of it 10 times and none of them predicted correctly, that's 0 in precision and a total recall (correct me here if necessary, thanks).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th and 14th cells of the Ipython notebook.

![image6]

For the first image, the model is relatively sure that this is an 80km/h sign (probability of 99%). The top five soft max probabilities were

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 99%                    | 80km/h |
| .000014%               | 50km/h |
| .000006%               | 60km/h |
| .0000000001%           | 100km/h |
| almost 0               | No passing for vehicles over 3.5 metric tons |

![image7]

Since it's perhaps the most easy sign of all, the model also does well on the second image. The top five softmax probabilities were

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 99%                    | Road work |
| .0071%               | Bicycles crossing |
| .0.0015%               | Wild animals crossing |
| .000035%           | Road narrows on the right |
| .000031%               | Turn right ahead |


![image8]

I thought the third image should be quite easy but the result doesn't agree. It does recognize the sign, with a lower probability compares with 30km/h. Here's the top five probabilities

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 55.75%                   | 30km/h |
| 39.83%               | Stop |
| 2.49%               | General caution |
| .809%           | 20km/h |
| .368%               | Yield |

![image9]

The model does well on predicting the fourth image also.

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 99%                    | Yield |
| .026%               | 50km/h |
| .0001%               | 30km/h |
| .00004%           | End of no passing by vehicles over 3.5 metric tons |
| .0000079%               | End of no passing |

![image10]

For the fifth image, it performed well even there's some chance to mistake it to General caution, but it's pretty sure that's a Traffic sign.

| Probability             |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 99%                   | Traffic signals |
| .061%               | General caution |
| .00018%               | Pedestrians |
| .000001%           | Road narrows on the right |
| .00000019%               | Keep right |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
![image2]
The above is the 18 features the first Convolutional layer captured for the downloaded Stop sign image. The 5x5 kernel generally detected these apparent form (shape) especially these in the middle. This is partly because no padding was given in convolving. Then a possible improvement would be to add some padding to help capture the image edges.

![image3]
The second layer's feature map is harder for a human to make sense of. I think it's because deeper Convolutional layer goes on to capture more tiny details (features) in a more micro scope. In general, the more details the deeper layer recognizes, the more confident the final classification predicts.
