input = readmatrix('train.csv');
target = input(:, 2);
input = input(:, 3:end);

target = OneHotEncoder(target');
input = reshape(input, [60000, 28, 28]);

parfor i = 1:length(input)
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
end
input = input / 255;
input = reshape(input, [60000, 784]);
target = target / 10;
[inputTrain, inputTest, targetTrain, targetTest] = SplitTrainTest(input, target);
% Map the scalar value to a one hot encoded vector to utilize softmax.
dim = [784 100 50 10];
functions = [NeuralNet.LOGSIG NeuralNet.LOGSIG NeuralNet.SOFTMAX];
learning = NeuralNet.MINIBATCH;
learningrate = 0.2;
batchSize = length(inputTrain) / 100;
nn = NeuralNet(dim, functions, learning, learningrate, batchSize);

EPOCHS = 10;
for i = 1:EPOCHS
   MSE = nn.train(inputTrain(1:batchSize,:)', targetTrain(1:batchSize,:)');
   fprintf('MSE at EPOCH: %d is: %d\n', i, MSE);
end

results = nn.test(inputTest');
