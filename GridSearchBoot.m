% Make sure to have run fashion MNIST before proceeding

INPUT_LAYER = 784;
OUTPUT_LAYER = 10;

LAST_FUNCTION = NeuralNet.SOFTMAX;

dims = {[INPUT_LAYER 225 OUTPUT_LAYER]};
        %[INPUT_LAYER 220 OUTPUT_LAYER]};
        %[INPUT_LAYER 230 OUTPUT_LAYER];
    %[INPUT_LAYER 275 OUTPUT_LAYER]};

functions = {[NeuralNet.SIG LAST_FUNCTION]};
    %[NeuralNet.SIG LAST_FUNCTION];
    %[NeuralNet.SIG LAST_FUNCTION];
    %[NeuralNet.SIG LAST_FUNCTION]};

alphas = [0.19 0.2 0.21];

[bestAccuracy, accuracies, bestNet] = GridSearch(dims, functions, alphas, inputTrain,...
    targetTrain, inputTest, targetTest, 200);

