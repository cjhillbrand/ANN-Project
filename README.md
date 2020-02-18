# ANN-Project
Store source code for the Back-propagation homework assignment and then the final project.

## Design of NeuralNet.m
This is the backbone of the backpropogation network. The NeuralNet.m is designed similar to the Sci-Kit learn model of constructing/training and testing a data-generated model.

### Public Methods
* Constructor: The constructor takes in an array of layers. The first element in this array describes the number of elements in the input layer and then every one following describes the hidden layer, and the last one describes the output layer. The second parameter is an array of transfer functions. There is a one to one mapping of transfer function to layer. The last parameter is choosing whether to have the network perform batch or on-line learning. So for example if someone was to call, `NeuralNet([3, 3, 1], ['sig, 'sig'], 'batch')` then a NeuralNet that has an input layer of 3 elements a hidden layer also containing 3 elements and an output layer of 1, that all have sigmoid transfer functions and the weights are updated every epoch would be constructed. 

* Train:  The train method takes in a set of input vectors and a set of target values. The train method uses forward propogation to find actual values, then depending on the learning rule (stochastic or batch) updates the weights and biases accordingly.

* Predict: The predict method takes in a set of input vectors and returns a set of target values. The target values are what were computed from forward propogation.

* InitWeights: Given a weight object fills sets the weights to of the network to the object passed. 

### Private Methods:
* Forward Propogation: Given an input value runs that value through the netwrok saving values such as the error, sensitivity, and actual value outputed. Returns the actual value returned from the network.

* Backward Propogation: Using the fields updated by the forward propogation adjusts the weights and biases of the network. This method once done updating zeros out any of the accumulated sums of errors, sensitivity, and stored actual values.

_Have to figure out other Private methods we need. Maybe a computeSensitivity? Computer MSE? Not sure... what we decide to do here will affect what we will store in our private fields._

_Look into cell array this can store matrices of unequal sizes_

### Private Fields:
* Weights: A 3 dimensional object that stores the weight matrix for each layer. 

* Biases: A matrix that stores the biases of the network.

_Depending on what info we need at each iteration for the batch learning we will store them in these private fields. Need to discuss with Ryan more in depth in what we actuall need_
