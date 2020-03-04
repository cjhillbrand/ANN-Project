input = readmatrix('train.csv');
target = input(:, 2);
input = input(:, 3:end);

target = OneHotEncoder(target');
input = reshape(input, [60000, 28, 28]);

parfor i = 1:length(input)
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SMOOTHING_KERNEL);
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
end
input = input / 255;
input = reshape(input, [60000, 784]);
[inputTrain, inputTest, targetTrain, targetTest] = SplitTrainTest(input, target);
% Map the scalar value to a one hot encoded vector to utilize softmax.
dim = [784 250 125 10];
functions = [NeuralNet.SIG NeuralNet.SIG NeuralNet.SOFTMAX];
learning = NeuralNet.MINIBATCH;
learningrate = 0.01;
batchSize = length(inputTrain) / 1000;
nn = NeuralNet(dim, functions, learning, learningrate, batchSize);

EPOCHS = 1;
for i = 1:EPOCHS
   MSE = nn.train(inputTrain', targetTrain');
   fprintf('MSE at EPOCH: %d is: %d\n', i, MSE);
end

results = nn.test(inputTest');
sum = 0;
results = setMax(results);
for i = 1:length(results)
   if (results(:, i) == targetTrain(i, :)')
      sum = sum + 1; 
   end
end
accuracy = sum / length(results); 
