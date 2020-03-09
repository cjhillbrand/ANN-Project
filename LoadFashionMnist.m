input = readmatrix('./Data/train.csv');
target = input(:, 2);
input = input(:, 3:end);

target = OneHotEncoder(target)';
input = reshape(input, [60000, 28, 28]);
inputSmooth2 = input;
inputNoSmooth = input;

parfor i = 1:length(input)
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SMOOTHING_KERNEL);
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
    inputSmooth2(i) = ImageHandler.applyKernel(inputSmooth2(i), ImageHandler.SMOOTHING_KERNEL, 2);
    inputSmooth2(i) = ImageHandler.applyKernel(inputSmooth2(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
    inputNoSmooth(i) = ImageHandler.applyKernel(inputNoSmooth(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
end

% Get rid of border pixels since they dont offer a lot of info.
input = input(:, 4:25, 4:25);
inputSmooth2 = inputSmooth2(:, 4:25, 4:25);
inputNoSmooth = inputNoSmooth(:, 4:25, 4:25);

input = reshape(input, [60000, 484])';
inputSmooth2 = reshape(inputSmooth2, [60000, 484])';
inputNoSmooth = reshape(inputNoSmooth, [60000, 484])';
input = [input inputSmooth2 inputNoSmooth];
target = [target target target];
input = zscore(input, 1, 'all');
