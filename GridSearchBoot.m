% Make sure to have run fashion MNIST before proceeding

INPUT_LAYER = 625;
OUTPUT_LAYER = 10;

LAST_FUNCTION = NeuralNet.SOFTMAX;

dims = {[INPUT_LAYER 225 OUTPUT_LAYER]};

functions = {[NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION]};

alphas = [0.1];
%alphas = [0.2 0.2 0.2 0.2 0.2];
%alphas = [alphas alphas];

[bestAccuracy, accuracies, bestNet] = GridSearch(dims, functions, alphas,...
    input, target, 100);

