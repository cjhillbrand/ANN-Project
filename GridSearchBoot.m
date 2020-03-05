% Make sure to have run fashion MNIST before proceeding

INPUT_LAYER = 784;
OUTPUT_LAYER = 10;

LAST_FUNCTION = NeuralNet.SOFTMAX;

dims = {[INPUT_LAYER 250 OUTPUT_LAYER];
    [INPUT_LAYER 250 OUTPUT_LAYER];
    [INPUT_LAYER 200 OUTPUT_LAYER];
    [INPUT_LAYER 200 OUTPUT_LAYER]};

functions = {[NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.LOGSIG LAST_FUNCTION];
    [NeuralNet.SIG LAST_FUNCTION];
    [NeuralNet.LOGSIG LAST_FUNCTION]};

alphas = [0.7 0.75 0.8];

[bestAccuracy, accuracies, bestNet] = GridSearch(dims, functions, alphas, inputTrain,...
    targetTrain, inputTest, targetTest);