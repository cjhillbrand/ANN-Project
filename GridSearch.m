function [bestAccuracy, accuracies, bestNet] = GridSearch(dims,...
    functions, alphas, inputs, targets, testInputs, testTargets)
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
    while (epoch < 10)
        epoch = epoch + 1;
        parfor i = 1:length(nn)
            nn{i}.train(inputs, targets);
            a = nn{i}.test(testInputs);
            accuracyTemp = sum(sum(a .* testTargets)) / length(a);
            fprintf('EPOCH: %d NN: %d Accuracy: %f\n', epoch, i, accuracyTemp);
            accuracies(i) = accuracyTemp;
        end
    end
    bestAccuracy = max(accuracies);
    bestNet = nn(bestAcccuracy == accuracies);
end