# 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
## Week 1
- Monitor the accuracy of the housing price predictions
- Analyze housing price predictions that come from a single layer neural network
- Use TensorFlow to build a single layer neural network for fitting linear models
## Week 2
- Use callback functions for tracking model loss and accuracy during training
- Make predictions on how the layer size affects network predictions and training speed
- Implement pixel value normalization to speed up network training
- Build a multilayer neural network for classifying the Fashion MNIST image dataset
## Week 3
- Use callback functions to interrupt training after meeting a threshold accuracy
- Test the effect of adding convolution and MaxPooling to the neural network for classifying Fashion MNIST images on classification accuracy
- Explain and visualize how convolution and MaxPooling aid in image classification tasks
## Week 4
- Reflect on the possible shortcomings of your binary classification model implementation
- Execute image preprocessing with the Keras ImageDataGenerator functionality
- Carry out real life image classification by leveraging a multilayer neural network for binary classification
# 2. Convolutional Neural Networks in TensorFlow
## Week 1
- Gain understanding about Keras’ utilities for pre-processing image data, in particular the ImageDataGenerator class
- Develop helper functions to move files around the filesystem so that they can be fed to the ImageDataGenerator
- Learn how to plot training and validation accuracies to evaluate model performance
- Build a classifier using convolutional neural networks for performing cats vs dogs classification
## Week 2
- Recognize the impact of adding image augmentation to the training process, particularly in time
- Demonstrate overfitting or lack of by plotting training and validation accuracies
- Familiarize with the ImageDataGenerator parameters used for carrying out image augmentation
- Learn how to mitigate overfitting by using data augmentation techniques
## Week 3
- Master the keras layer type known as dropout to avoid overfitting
- Achieve transfer learning in code using the keras API
- Code a model that implements Keras’ functional API instead of the commonly used Sequential model
- Learn how to freeze layers from an existing model to successfully implement transfer learning
- Explore the concept of transfer learning to use the convolutions learned by a different model from a larger dataset
## Week 4
- Build a multiclass classifier for the Sign Language MNIST dataset
- Learn how to properly set up the ImageDataGenerator parameters and the model definition functions for multiclass classification
- Understand the difference between using actual image files vs images encoded in other formats and how this changes the methods available when using ImageDataGenerator
- Code a helper function to parse a raw CSV file which contains the information of the pixel values for the images used
# 3. Natural Language Processing in TensorFlow
# 4. Sequences, Time Series and Prediction
## Week 1
C4_W1_Lab_1_time_series.ipynb
- Trend: The trend describes the general tendency of the values to go up or down as time progresses. 
Given a certain time period, you can see if the graph is following an upward/positive trend, downward/negative trend, or just flat. 
For instance, the housing prices in a good location can see a general increase in valuation as time passes.
- Seasonality: This refers to a recurring pattern at regular time intervals. 
For instance, the hourly temperature might oscillate similarly for 10 consecutive days and 
you can use that to predict the behavior on the next day.
- Noise: In practice, few real-life time series have such a smooth signal. They usually have some noise riding over that signal. 
- Autocorrelation: Time series can also be autocorrelated. This means that measurements at a given time step is a function of previous time steps.
- Non-stationary Time Series: It is also possible for the time series to break an expected pattern. 
As mentioned in the lectures, big events can alter the trend or seasonal behavior of the data. 

C4_W1_Lab_2_forecasting.ipynb
- Moving Average: One technique you can use is to do a moving average. 
This sums up a series of time steps and the average will be the prediction for the next time step. 
- Differencing: Since the seasonality period is 365 days, you will subtract the value at time t – 365 from the value at time t. 
- Smoothing: You can use the same ```moving_average_forecast()``` function to smooth out past values before adding them back to the time differenced moving average.

