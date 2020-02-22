functions = [NeuralNet.SIG NeuralNet.SIG];
type = NeuralNet.BATCH;
a = 0.1;
ZERO = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]';
ONE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
TWO = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]';

ZERO_t = [1; 0; 0];
ONE_t = [0; 1; 0];
TWO_t = [0; 0; 1];

ZERO_NOISE_4 = addNoise(ZERO, 4);
ZERO_NOISE_8 = addNoise(ZERO, 8);
ONE_NOISE_4 = addNoise(ONE, 4);
ONE_NOISE_8 = addNoise(ONE, 8);
TWO_NOISE_4 = addNoise(TWO, 4);
TWO_NOISE_8 = addNoise(TWO, 8);

p = [ZERO ,ONE, TWO];
t = [ZERO_t ONE_t TWO_t];

EPOCHS = 50000;

dim = [length(ZERO) 5 length(t)];
nn = NeuralNet(dim, functions, type, a);

% Plot section
x = 1:EPOCHS;
y = zeros(1, EPOCHS);

for i = 1:EPOCHS
    MSE = nn.train(p, t);
    %y(i) = MSE;
    a = nn.test(p);
end

figure;
xlim([0 50000]);
ylim([0 2]);
plot(x, y);