input = readmatrix('train.csv');
target = input(:, 2);
input = input(:, 3:end);

target = OneHotEncoder(target)';
input = reshape(input, [60000, 28, 28]);

parfor i = 1:length(input)
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SMOOTHING_KERNEL);
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
end
input = zscore(input, 1, 'all');
input = reshape(input, [60000, 784])';
[inputTrain, inputTest, targetTrain, targetTest] = SplitTrainTest(input, target);

%% One set of HyperParamaters
% dim = [784 128 10];
% functions = [NeuralNet.SIG NeuralNet.SOFTMAX];
% learning = NeuralNet.ONLINE;
% learningrate = 0.4;
% batchSize = length(inputTrain) / 2000;
% 
% nn = NeuralNet(dim, functions, learning, learningrate, batchSize);
% 
% EPOCHS = 10;
% for i = 1:EPOCHS
%    MSE = nn.train(inputTrain, targetTrain);
%    fprintf('MSE at EPOCH: %d is: %d\n', i, MSE);
% end
% 
% results = nn.test(inputTest);
% sum = 0;
% results = setMax(results);
% for i = 1:length(results)
%    if (results(:, i) == targetTest(:, i))
%       sum = sum + 1; 
%    end
% end
% accuracy = sum / length(results); 
