functions = [NeuralNet.SIG NeuralNet.SIG];
type = NeuralNet.BATCH;
a = 0.1;
ZERO = [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0]';
ONE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';
TWO = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]';

ZERO_t = [1; 0; 0];
ONE_t = [0; 1; 0];
TWO_t = [0; 0; 1];

noise = [0 4 8];

p = [ZERO ,ONE, TWO];
t = [ZERO_t ONE_t TWO_t];

EPOCHS = 50000;

dim = [length(ZERO) 5 length(t)];
nn = NeuralNet(dim, functions, type, a);

% Plot section
x = 1:EPOCHS;
y = zeros(1, EPOCHS);
results = zeros(3, EPOCHS);
for i = 1:EPOCHS
    MSE = nn.train(p, t);
    y(i) = MSE;
    
    % Uncomment this part for the graphing of accuracy also uncomment the
    % Graph stuff down below. It just takes a long time to run with a lot
    % of epochs
%     for k = 1:length(noise)
%         for j = 1:10
%            input = [addNoise(ZERO, noise(k)) addNoise(ONE, noise(k)) addNoise(TWO, noise(k))];
%            a = nn.test(input);
%            a = setMax(a);
%            results(k, i) = results(k, i) + 3 - sum(sum((t - a) == -1));
%         end
%     end
    
    a = nn.test(p);
end
% results = results * 100 / 30; % Make into percent correct
% figure;
% xlim([0 EPOCHS]);
% ylim([0 100]);
% plot(x, results(1, :));
% title('Percent Correct over Epochs with No Noise');
% xlabel('EPOCHS');
% ylabel('Number of Correct Matches over Total');
% saveas(gcf, 'NoNoise.png');
% 
% figure;
% xlim([0 EPOCHS]);
% ylim([0 100]);
% plot(x, results(2, :));
% title('Percent Correct over Epochs with 4 Pixels Flipped');
% xlabel('EPOCHS');
% ylabel('Number of Correct Matches over Total');
% saveas(gcf, '4Noise.png');
% 
% figure;
% xlim([0 EPOCHS]);
% ylim([0 100]);
% plot(x, results(3, :));
% title('Percent Correct over Epochs with 8 Pixels Flipped');
% xlabel('EPOCHS');
% ylabel('Number of Correct Matches over Total');
% saveas(gcf, '8Noise.png');
