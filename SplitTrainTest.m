function [trainInput, trainTarget, testInput, testTarget] = SplitTrainTest(inputs, targets)
    [~, n] = size(A);
    P = 0.7;
    idx = randperm(n);
    
    trainInput = inputs(:, idx(1:round(P*n)));
    trainTarget = targets(:, idx(1:round(P*n)));
    
    testInput = inputs(:, idx(round(P*n) + 1:n));
    testTarget = targets(:, idx(round(P*n) + 1:n));
    
end