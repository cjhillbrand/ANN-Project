% function train(obj, p, t)
% function initWeights(obj, weights)
% function actual = test(obj, p)
% function obj = NeuralNet(dimensions, transFuncs, learning, learningRate)
hidden = [2 3 4 5 6 7 8 9 10];
functions = [NeuralNet.SIG NeuralNet.SIG];
type = NeuralNet.BATCH;
a = 0.1;
ZERO = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]';
ONE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
TWO = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]';


p = [ZERO ,ONE, TWO];
t = [1, 0, 0; 0, 1, 0; 0, 0, 1];

dim = [length(ZERO) 5 length(t)];

nn = NeuralNet(dim, functions, type, a);
EPOCHS = 50000;
MSE = 2;
i = 0;

dim = [length(ZERO) 5 length(t)];
nn = NeuralNet(dim, functions, type, a);

while MSE > 0.01 && i < EPOCHS
MSE = nn.train(p, t);
i = i + 1;
    if (mod(i, 1000) == 0)
       fprintf('Error at iteration %d is %f \n', i, MSE); 
    end
end

fprintf('Error at iteration %d is %f \n', i, MSE);

% for i = 1:EPOCHS
%    MSE = nn.train(p, t); 
%     fprintf('ERROR IS: %d\n', MSE);
% end

a = nn.test(p);
