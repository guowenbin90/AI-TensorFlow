# AI-TensorFlow
**Artificial Intelligence:** A field of computer science that aims to make computers achieve human-style intelligence. There are many approaches to reaching this goal, including machine learning and deep learning.

**Machine Learning:** A set of related techniques in which computers are trained to perform a particular task rather than by explicitly programming them.

**Neural Network:** A construct in Machine Learning inspired by the network of neurons (nerve cells) in the biological brain. Neural networks are a fundamental part of deep learning, and will be covered in this course.

**Deep Learning:** A subfield of machine learning that uses multi-layered neural networks. Often, “machine learning” and “deep learning” are used interchangeably.


## Lesson 2: Introduction to Machine Learning
The training process (happening in ```model.fit(...)```) is really about tuning the internal variables of the networks to the best possible values, 
so that they can map the input to the output. A optimization process called Gradient Descent.   
The value of the loss is calculated using a loss function, which we specified with the loss parameter when calling ```model.compile()```.

[Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

**Epochs:** One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.  
**Batch Size:** Total number of training examples present in a single batch.  
**Iterations:** Iterations is the number of batches needed to complete one epoch.  

[Google’s machine learning crash course](https://developers.google.com/machine-learning/crash-course/reducing-loss/video-lecture)  
**Feature:** The input(s) to our model  
**Examples:** An input/output pair used for training  
**Labels:** The output of the model  
**Layer:** A collection of nodes connected together within a neural network.  
**Model:** The representation of your neural network  
**Dense and Fully Connected (FC):** Each node in one layer is connected to each node in the previous layer.  
**Weights and biases:** The internal variables of model  
**Loss:** The discrepancy between the desired output and the actual output  
**MSE:** Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.  
**Gradient Descent:** An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.  
**Optimizer:** A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for ADAptive with Momentum. It is considered the best-practice optimizer.)  
**Learning rate:** The “step size” for loss improvement during gradient descent.  
**Batch:** The set of examples used during training of the neural network  
**Epoch:** A full pass over the entire training dataset  
**Forward pass:** The computation of output values from input  
**Backward pass (backpropagation):** The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.  

## Lesson 3: First Model - Fashion MNIST
**Flattening:** The process of converting a 2d image into 1d vector  
**ReLU:** An activation function that allows a model to solve nonlinear problems  
**Softmax:** A function that provides probabilities for each possible output class  
**Classification:** A machine learning model used for distinguishing among two or more output categories  

|    |   Classification      |  Regression |
|:----------|:-------------|:------|
| Output |  List of numbers that represent probabilities for each class | Single number |
| Example | Fashion MNIST | Celsius to Fahrenheit |
| Loss | Sparse categorical crossentropy | Mean squared error |
| Last Layer Activation Function | Softmax | None |

## Lesson 4: Introduction to CNNs
A convolution is the process of applying a filter (“kernel”) to an image. Max pooling is the process of reducing the size of the image through downsampling.  
**CNNs:** Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers.  
**Convolution:** The process of applying a kernel (filter) to an image  
**Kernel / filter:** A matrix which is smaller than the input, used to transform the input into chunks  
**Padding:** Adding pixels of some value, usually 0, around the input image  
**Pooling:** The process of reducing the size of an image through downsampling.There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.  
**Maxpooling:** A pooling process in which many values are converted into a single value by taking the maximum value from among them.  
**Stride:** the number of pixels to slide the kernel (filter) across the image.  
**Downsampling:** The act of reducing the size of an image  
More details about [CNN](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) 

## Lesson 5: Going Further with CNNs
activation='softmax' and 'sigmoid' work well in a binary classification problem. 

activation='softmax'
```
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
activation='sigmoid' 
```
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
### Three different techniques to prevent overfitting:
**Early Stopping:** In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.  
**Image Augmentation:** Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.  
**Dropout:** Removing a random selection of a fixed number of neurons in a neural network during training.  
[Memorizing is not learning! — 6 tricks to prevent overfitting in machine learning](https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42)

## Lesson 6: Transfer Learning
Using Transfer Learning to create very powerful Convolutional Neural Networks with very little effort  
**Transfer Learning:** A technique that reuses a model that was created by machine learning experts and that has already been trained on a large dataset. When performing transfer learning we must always change the last layer of the pre-trained model so that it has the same number of classes that we have in the dataset we are working with.  
**Freezing Parameters:** Setting the variables of a pre-trained model to non-trainable. By freezing the parameters, we will ensure that only the variables of the last classification layer get trained, while the variables from the other layers of the pre-trained model are kept the same.  
**MobileNet:** A state-of-the-art convolutional neural network developed by Google that uses a very efficient neural network architecture that minimizes the amount of memory and computational resources needed, while maintaining a high level of accuracy. MobileNet is ideal for mobile devices that have limited memory and computational resources.  

## Lesson 7: Saving and Loading Models

## Lesson 8: Time Series Forecasting
Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow

neural networks rely on stochasticity (i.e. randomness) to initialize their parameters and gradient descent selects random batches of training data at each iteration  
RNN  
LSTM  
CNN
## Lesson 9: NLP: Tokenization and Embeddings
Natural Language Processing, or NLP for short, focuses on analyzing text and speech data. This can range from simple recognition (what words are in the given text/speech), to sentiment analysis (was a review positive or negative), and all the way to areas like text generation (creating novel song lyrics from scratch).

Embeddings are clusters of vectors in multi-dimensional space, where each vector represents a given word in those dimensions. 
## Lesson 10: NLP: Recurrent Neural Networks
Simple RNNs are not always enough when working with text data. Longer sequences, such as a paragraph, often are difficult to handle, as the simple RNN structure loses information about previous inputs fairly quickly.

Long Short-Term Memory models, or LSTMs, help resolve this by keeping a “cell state” across time. These include a “forget gate”, where the cell can choose whether to keep or forget certain words to carry forward in the sequence.

Another interesting aspect of LSTMs is that they can be bidirectional, meaning that information can be passed both forward (later in the text sequence) and backward (earlier in the text sequence).
```
#Two bidirectional LSTM layers with 64 nodes each
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True)
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
```
LSTMs vs. Convolutions vs. GRUs  
- **LSTMs:** Utilizes “forget” and “learn” gates that feed to “remember” and “use” gates, where remembering is for further storage for the next input, and using is for generating the current output.
- **Convolutions:** Utilizes “filters” that can slide over multiple words at a time and extract features from those sequences of words. Can be used for purposes other than a recurrent neural network.
- **GRUs:** Utilizes “update” and “reset” gates, where the “update” gate determines updates to the existing stored knowledge, and the reset gate determines how much to forget in the existing stored knowledge.

Text generation can be done through simply predicting the next most likely word, given an input sequence. This can be done over and over by feeding the original input sequence, plus the newly predicted end word, as the next input sequence to the model. As such, the full output generated from a very short original input can effectively go on however long you want it to be.  
N-Grams used in the pre-processing work - a single input sequence might actually become a series of sequences and labels.
## Lesson 11: Introduction to TensorFlow Lite
Deploy deep learning models on mobile and embedded devices with TensorFlow Lite.  
TF Lite on AndroidTF, iOS with Swift and IoT
