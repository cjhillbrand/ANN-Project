dims = [numRows * numCols 5 1];
functions = [NeuralNet.SIG NeuralNet.SIG];
learning = NeuralNet.MINIBATCH;
alpha = 0.5;
batchSize = length(labels) / 1000;
nn = NeuralNet(dims, functions, learning, alpha, batchSize);

EPOCHS = 10;


x = 1:EPOCHS;
y = zeros(1, EPOCHS);
for i = 1:EPOCHS
   MSE = nn.train(images, labels);
   y(i) = MSE;
   fprintf('MSE: %d \n', MSE);
end

predicted = nn.test(testImages);
MSE = (testLabels - predicted)' * (testLabels - predicted) / length(testLabels);

figure;
xlim([0, EPOCHS]);
ylim([0 2])
plot(x, y);
title('Mean Squared Error over Epochs');
xlabel('EPOCHS');
ylabel('Mean Squared Error');
%saveas(gcf, 'MSEMINIBATCH.png');