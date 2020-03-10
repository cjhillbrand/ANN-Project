input = readmatrix('./Data/test.csv');
input = input(:, 2:end);

%target = OneHotEncoder(target)';
input = reshape(input, [10000, 28, 28]);
inputSmooth2 = input;
inputNoSmooth = input;
parfor i = 1:length(input)
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SMOOTHING_KERNEL);
    input(i) = ImageHandler.applyKernel(input(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
    inputSmooth2(i) = ImageHandler.applyKernel(inputSmooth2(i), ImageHandler.SMOOTHING_KERNEL, 2);
    inputSmooth2(i) = ImageHandler.applyKernel(inputSmooth2(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
%     inputNoSmooth(i) = ImageHandler.applyKernel(inputNoSmooth(i), ImageHandler.SECOND_GRADIENT_KERNEL_X_Y);
end
input = input(:, 3:27, 3:27);
inputSmooth2 = inputSmooth2(:, 3:27, 3:27);

input = reshape(input, [10000, 625])';
inputSmooth2 = reshape(inputSmooth2, [10000, 625])';
% inputNoSmooth = reshape(inputNoSmooth, [10000, 784])';
input = [input inputSmooth2];

inputSub = normalize(input);
