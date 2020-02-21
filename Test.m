% function train(obj, p, t)
% function initWeights(obj, weights)
% function actual = test(obj, p)
% function obj = NeuralNet(dimensions, transFuncs, learning, learningRate)
dim = [2 3 1];
functions = [NeuralNet.SIG NeuralNet.SIG];
type = NeuralNet.ONLINE;
a = 0.05;

p = [1, 3, -1, 0; 2, 1, 4, 1 ];
t = [0, 1, 2, 3];
nn = NeuralNet(dim, functions, type, a);

nn.train(p, t);
a = nn.test(p);
a