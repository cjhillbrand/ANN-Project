% Make sure to have run fashion MNIST before proceeding

INPUT_LAYER = 784;
OUTPUT_LAYER = 10;

LAST_FUNCTION = NeuralNet.SOFTMAX;

dims = {[INPUT_LAYER 225 OUTPUT_LAYER]};
        %[INPUT_LAYER 50 25 OUTPUT_LAYER]};
        %[INPUT_LAYER 40 OUTPUT_LAYER]};

functions = {[NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION]};

alphas = [0.2 0.2 0.2 0.2 0.2];
%alphas = [alphas alphas];

[bestAccuracy, accuracies, bestNet] = GridSearch(dims, functions, alphas,...
    input, target, 1000);

