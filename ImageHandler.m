% A Static class that handles the manipulation of images.
% This class handles the smoothing and gradient finding of images with
% pre-defined kernels.
classdef ImageHandler
    
    % These are the pre-defined kernels for image processing.
    % The kernels are defined to the size that best fits the MNIST
    % fashion data set which has relatively low resolution.
    properties (Constant)
        % Apply to produce the gradient image in X direction
        GRADIENT_KERNEL_X = [-1 0 1; -2 0 2; -1 0 1];
        
        % Apply to produce the gradient image in Y direction
        GRADIENT_KERNEL_Y = [-1 -2 -1; 0 0 0; 1 2 1];
        
        % Apply to produce Second gradient with Respect to X and Y 
        SECOND_GRADIENT_KERNEL_X_Y = [1 0 -1; 0 0 0; -1 0 1]; 
        
        % Apply to produce a smoothed image using a gaussian blur
        SMOOTHING_KERNEL = [1/16 2/16 1/16; 2/16 4/16 2/16; 1/16 2/16 1/16];       
    end
    
    methods(Static)
        % Applies the given kernel to the image. The convolution is set up
        % so that the image returned is of the same size of the image.
        function result = applyKernel(image, kernel, iteration)
            % Checks to make sure that the kernel passed in is valid.
            if (~isequal(kernel, ImageHandler.GRADIENT_KERNEL_X) &&...
                    ~isequal(kernel, ImageHandler.GRADIENT_KERNEL_Y) &&...
                    ~isequal(kernel, ImageHandler.SECOND_GRADIENT_KERNEL_X_Y) &&...
                    ~isequal(kernel, ImageHandler.SMOOTHING_KERNEL))
                error('Must pass in pre-defined kernel');
            end
            if (size(image) < 3)
                error('Image must be larger than kernel');
            end
            result = conv2(image, kernel, 'same');
            if exist('iteration', 'var')
                for i = 1:iteration
                    result = conv2(result, kernel, 'same');
                end    
            end
        end
    end
    
end
