# ANN-Project
Store source code for the Back-propagation homework assignment and then the final project.

## Design of NeuralNet.m
This is the backbone of the backpropogation network. The NeuralNet.m is designed similar to the Sci-Kit learn model of constructing/training and testing a data-generated model.

### Public Methods
* Constructor: The constructor takes in an array of layers. The first element in this array describes the number of elements in the input layer and then every one following describes the hidden layer, and the last one describes the output layer. The second parameter is an array of transfer functions. There is a one to one mapping of transfer function to layer. The last parameter is choosing whether to have the network perform batch or on-line learning. So for example if someone was to call, `NeuralNet([3, 3, 1], ['sig, 'sig'], 'batch')` then a NeuralNet that has an input layer of 3 elements a hidden layer also containing 3 elements and an output layer of 1, that all have sigmoid transfer functions and the weights are updated every epoch would be constructed. 

* Train:  The train method takes in a set of input vectors and a set of target values. The train method uses forward propogation to find actual values, then depending on the learning rule (stochastic or batch) updates the weights and biases accordingly.

* Predict: The predict method takes in a set of input vectors and returns a set of target values. The target values are what were computed from forward propogation.

## Future Design/Optimizations.
1. Include and research more transfer functions. Add these trasnfer functions and there derivatives to the NeuralNet.m class.
2. Create a function that takes in a set of hyperparamaters and tries all combinations of hyper paramaters. (Make sure to implement with a `parfor` loop. How should we measure this? MSE? Expand into different performance metrics down the line.
3. Given an image and a kernel such that the `kernel_rows <= Image_rows && kernel_cols <= Image_cols` applies a convolution to the image. (This is used for smoothing, gradients, derivatives, and second derivatives of images)
4. Create a scripting file to run multiple networks in parallel (This compliments the above point)\
5. Include more preprocessing on the image (away from convolution)
6. Transfer private fields in NeuralNet.m to use GPU arrays, also need to change input vectors (internally to GPU arrays)
7. Variable learning rates: Momentum, re-evaluate each epoch.
8. Stop short (prevent overfit)
#### Seperation of Work

| Ryan      | CJ      |
|-----------|---------|
| 2,5,6,7,8 | 1,2,3,5 |
