# Cifar-10 Classification

### Classification

A classification problem is when the output variable is a category, such as “red” or “blue” or “disease” and “no disease”. 
A classification model attempts to draw some conclusion from observed values. Given one or more inputs a classification model will try to predict the value of one or more outcomes.
We would be using CNN(Convolution Neural Networks) to do the same task.

#### CNN
Convolutional Neural Networks (CNNs) is the most popular neural network model being used for image classification problem. 
The big idea behind CNNs is that a local understanding of an image is good enough. 
The practical benefit is that having fewer parameters greatly improves the time it takes to learn as well as reduces the amount of data required to train the model.
Instead of a fully connected network of weights from each pixel, a CNN has just enough weights to look at a small patch of the image.

### CIFAR Dataset
The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. 
It is one of the most widely used datasets for machine learning research.The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes.
The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

# Defining the model architecture Using ConVnets


In the first stage, Our net will learn 32 convolutional filters, each of which with a 3 x 3 size. The output dimension is the same one of the input shape, so it will be 32 x 32 and activation is relu, which is a simple way of introducing non-linearity; followed by another 32 convolutional filters, each of which with a 3 x 3 size and activation is also relu. After that we have a max-pooling operation with pool size 2 x 2 and a dropout at 25%.

In the next stage in the deep pipeline, Our net will learn 64 convolutional filters, each of which with a 3 x 3 size. The output dimension is the same one of the input shape and activation is relu; followed by another 64 convolutional filters, each of which with a 3 x 3 size and activation is also relu. After that we have a max-pooling operation with pool size 2 x 2 and a dropout at 25%.

And the Final stage in the deep pipeline is a dense network with 512 units and relu activation followed by a dropout at 50% and by a softmax layer with 10 classes as output, one for each category.