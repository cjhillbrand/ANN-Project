dims = [numRows * numCols 5 1];
functions = [NeuralNet.SIG NeuralNet.SIG];
learning = NeuralNet.MINIBATCH;
alpha = 0.5;
batchSize = length(labels) / 1000;
nn = NeuralNet(dims, functions, learning, alpha, batchSize);
MSE = 10;
i = 0;

while MSE > 0.01
   MSE = nn.train(images, labels);
    %if (mod(i, 10) == 0)
    fprintf('ERROR %d\n', MSE);
    %end
end

predicted = nn.test(testImages);
MSE = (testLabels - predicted)' * (testLabels - predicted) / length(testLabels)