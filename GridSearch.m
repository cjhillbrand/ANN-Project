function [bestAccuracy, accuracies, bestNet] = GridSearch(dims,...
    functions, alphas, inputs, targets, epochs)
    nn = {};
    % Try different dims 
    for i = 1:length(dims)
        for j = 1:length(alphas)
            nn{(i - 1) * length(alphas) + j} = NeuralNet(dims{i}, functions{i},...
                NeuralNet.ONLINE, alphas(j));
        end
    end
    accuracies = zeros(length(nn), 1);

    epoch = 0;
    bestAccuracy = 0;
    [inputTrain, inputTest, targetTrain, targetTest] = SplitTrainTest(inputs, targets, 0.75);
    while (epoch < epochs)
        epoch = epoch + 1;
        for i = 1:length(nn)
            nn{i}.train(inputTrain, targetTrain);
            a = nn{i}.test(inputTest);
            accuracyTemp = sum(sum(setMax(a) .* targetTest)) / length(a);
            fprintf('EPOCH: %d NN: %d Accuracy: %f\n', epoch, i, accuracyTemp);
            accuracies(i) = accuracyTemp;
            if (accuracyTemp > bestAccuracy)
                bestAccuracy = accuracyTemp;
                bestNet = nn{i}.copy();
            end
            if (mod(epoch, 10) == 0)
               bestNet.writeNetToFile(['./Nets/Net_' num2str(epoch, '%d') '_Accuracy' num2str(round(bestAccuracy * 100), '%d') '.txt']) 
            end
        end
    end
end