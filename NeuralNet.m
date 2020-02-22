classdef NeuralNet < handle
    % NeuralNet class is a container for all of the hyper
    % paramaters that a user would want to tweak in a neural net
    % and not worry about the underlying weights, biases, and 
    % algorithms that are used to tweak these paramaters. 
    
    properties (Access=private)
        W %  The Cell object that holds all of the weight matrices
          %  for each layer of the neural network.
        b %  The Cell object that stores all of the vectors for each
          %  layer of the neural network.
        s % For batch learning stores the accumulation of the sesitivity
             % over a training set.
<<<<<<< HEAD
        sa % For batch learning stores the accumulation of the sensitivity
           % times a' from the previous output. (Refer to updating weights 
           % batch learning equation)
=======
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
        alpha % set the learning rate
        layers % Number of layers in the network.
        transFuncs % A record of all of the transfer functions.
        learning % Type of learning for this network.
    end
    

    
    % Define transfer function strings, and learning strings.
    properties (Constant)
        % For every transfer function we need to code a private function
        % for its actual value and its derivation.
        
        SIG = 0; PURELIN = 1; LOGSIG = 2;
        TRANS_FUNC_LOW_BOUND = 0; TRANS_FUNC_HIGH_BOUND = 2;
        % These are the three learning types that are supported.
        ONLINE = 10; BATCH = 11; MIN_BATCH = 12;
        LEARNING_LOW_BOUND = 10; LEARNING_HIGH_BOUND = 12;
        
        
    end
    
    %  Public Methods
    %  The high level access methods that a user can call to construct
    %  train and test the network created by the constructor.
    methods
        % The first argument is an array of ints that define the number
        % of elements each layer should predict.
        % The second argument is an array of integers that define the type
        % of trans function such that there are layers -1 elements in the
        % array.
        % The third argument is a string that denotes which type of
        % learning should be applied to the network.
        function obj = NeuralNet(dimensions, transFuncs, learning, ...
                learningRate)
            % Create weight cell matrix, create bias matrix, etc.
            layers = length(dimensions);
            
            % Check to make sure there are layers - 1 transfunctions
            if (layers ~= length(transFuncs) + 1)
                error('ERROR: There must be layers - 1 Transformation functions')
            end
            
            % Start off easy and check each function is a valid function
            % type.
           if (any(transFuncs(:) < NeuralNet.TRANS_FUNC_LOW_BOUND)...
                   || any(transFuncs(:) > NeuralNet.TRANS_FUNC_HIGH_BOUND))
            error('ERROR: Must pass in legal Tranformation Functions')
           end
            
            % Check that the learning type is also valid
            if (learning < NeuralNet.LEARNING_LOW_BOUND ||...
                    learning > NeuralNet.LEARNING_HIGH_BOUND)
                error('ERROR: Must pass in legal Learning Type')
            end
            
            % Done with simple Error checking, moveing on to creating the
            % cell arrays that contain the dimensions of our weights and
            % biases
            obj.W = {};
<<<<<<< HEAD
            obj.W{1} = (rand(dimensions(2), dimensions(1)));
            obj.b = {}; 
            obj.b{1} = rand(dimensions(2), 1);
            obj.s = {};
            obj.s{1} = zeros(dimensions(2), 1);
            obj.sa = {};
            obj.sa{1} = zeros(dimensions(2), dimensions(1));
            for i = 3:layers
               obj.W{i - 1} = (rand(dimensions(i), dimensions(i - 1)));
               obj.sa{i - 1} = zeros(dimensions(i), dimensions(i -1));
               obj.b{i - 1} = rand(dimensions(i), 1);
=======
            obj.W{1} = zeros(dimensions(2), dimensions(1));
            obj.b = {}; 
            obj.b{1} = zeros(dimensions(1), 1);
            obj.s = {};
            obj.s{1} = zeros(dimensions(1), 1);
            for i = 3:layers
               obj.W{i - 1} = zeros(dimensions(i), dimensions(i -1));
               obj.b{i - 1} = zeros(dimensions(i), 1);
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
               obj.s{i - 1} = zeros(dimensions(i), 1);
            end
            
            % Store the transformation functions.
            obj.transFuncs = transFuncs;
            
            % Store the learning type.
            obj.learning = learning;
            
            % Store the number of layers
            obj.layers = layers;
            
            % Set learning rate 
            obj.alpha = learningRate;
<<<<<<< HEAD
                        
=======
            
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
        end
        
        % Takes in a set of input vectors that have the same number of
        % elements as the first layer of the network. Takes in a set of t
        % vectors that match the number of elements of the output layer.
        % Also the number of p vectors and number of t vectors must be the
        % same. Trains the network with forward propogation and back
        % propogation. Updates the weights according to the learning type.
<<<<<<< HEAD
        function MSE = train(obj, p, t)
            Q = size(p, 2);
            
            MSE = 0;
            for i = 1:Q
                [n, a] = forwardPropogation(obj, p(:, i));
                sens = computeSensitivity(obj, n, a, t(:, i));
                MSE = MSE + (t(:, i) - a)' * (t(:, i) - a);
                if (obj.learning == obj.BATCH)
                   for j = length(obj.s)
                       obj.s{j} = obj.s{j} + sens{j};
                       obj.sa{j} = obj.sa{j} + sens{j} * evaluateFunc(obj, n{j}, obj.transFuncs(j))'; 
                   end
                end
                if (obj.learning == obj.ONLINE) 
                    updateWeights(obj, n, sens, p(:, i));
                end
            end
            if (obj.learning == obj.BATCH) 
               updateWeightsBatch(obj, Q); 
            end
            MSE = MSE / Q;
=======
        function train(obj, p, t)
            for i = 1:size(p, 2)
                [n, a] = forwardPropogation(obj, p);
                sens = computeSensitivity(obj, n, a, t);
                if (obj.learningType == obj.BATCH)
                   % increment global sensitivity 
                end
                if (obj.learningType == obj.ONLINE) 
                    updateWeights();
                end
            end
            if (obj.learningType == obj.BATCH) 
               updateWeights(); 
            end
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
        end
        
        % Takes in a set of input vectors that must match the number of
        % elements in the input layer of the network. Returns a set of
        % vectors that represent the actual values produced from the test
        % inputs.
        function a = test(obj, p)
            Q = size(p, 2);
            a = zeros(length(obj.b{obj.layers - 1}), Q);
            for i = 1:Q
                [~, a(:, i)] = forwardPropogation(obj, p(:, i));
            end
        end
        
    end
    
    methods (Access=private)
        
        % Not sure what these functions need just yet. They do need to
        % store values differently for different types of learning.
        function [nCell, a] = forwardPropogation(obj, p)
            nCell = {};
<<<<<<< HEAD
            n = obj.W{1} * p + obj.b{1};
            nCell{1} = n; 
            a = evaluateFunc(obj, n, obj.transFuncs(1)); 
            for m = 2: obj.layers - 1
               n = obj.W{m} * a + obj.b{m};
               nCell{m} = n;
               a = evaluateFunc(obj, n, obj.transFuncs(m));
            end
=======
            n = obj.w{1} * p + obj.b{1};
            nCell{1} = n; 
            a = evaluateFunc(obj, n, obj.transFuncs(1)); 
            for m = 2:obj.layers - 1
               n = obj.w{m} * a + obj.b{m};
               nCell{m} = n;
               a = evaluateFunc(obj, n, obj.transFuncs(m));
            end
            
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
        end
        
        function sens = computeSensitivity(obj, n, a, t)
            sens = {};
<<<<<<< HEAD
            jacob = evaluateJacob(obj, n{obj.layers - 1}, obj.transFuncs(obj.layers - 1));
            sens{1} = -2 * jacob * (t - a);
            for i = obj.layers - 2 : -1 : 1
                deriv = evaluateJacob(obj, n{i}, obj.transFuncs(i));
                sens = [deriv * obj.W{i + 1}' * sens{1}, sens];
            end
        end
        
        function updateWeights(obj, n, S, p)
%             fprintf('Sensitivity:');
%             obj.b{1}
            obj.W{1} = obj.W{1} - obj.alpha * S{1} * p';
            obj.b{1} = obj.b{1} - obj.alpha * S{1};
            
            for i = 2 : obj.layers - 1
                obj.W{i} = obj.W{i} - obj.alpha * S{i} * evaluateFunc(obj, n{i-1}, obj.transFuncs(i))';
                obj.b{i} = obj.b{i} - obj.alpha * S{i};
            end
        end
        
        function updateWeightsBatch(obj, Q)
            for i = 1:obj.layers - 1
               obj.W{i} = obj.W{i} - obj.alpha / Q * obj.sa{i}; 
               obj.b{i} = obj.b{i} - obj.alpha / Q * obj.s{i};
=======
            sens = [-2 * (t - a{obj.layers -1}) *...
                evaluateDeriv(obj, n{obj.layers -1}), sens];
            for i = obj.layers -1 : -1 : 1
                sens = [evaluateDeriv(obj, n{obj.layers - 1} *...
                    obj.W{i + 1}' * sens{1}), sens];
            end
        end
        
        function result = updateWeights(obj, n, m)
            if (m == 0)
                obj.W{m} = obj.W{m} - obj.alpha * S{m} * p;
                obj.b{m} = obj.b{m} - obj.alpha * S{m};
            else
                obj.W{m} = obj.W{m} - obj.alpha * S{m} * evaluateFunc(n{m-1});
                obj.b{m} = obj.b{m} - obj.alpha * S{m};
>>>>>>> d8d109d1c5553672f14a80f659b479d8b89a50c6
            end
        end
        
        % Given a value evaluates that value at the type of function
        % also passed through.
        function result = evaluateFunc(obj, value, type)
            % Example for the sigmoid function. Must have
            % an if/equation for each type of transfer function.
            result = zeros(length(value), 1);
            for i = 1:length(value)
                if (type == obj.SIG)
                    result(i) = 1 / (1 + exp(-value(i)));
                end
                if (type == obj.LOGSIG)
                   result(i) = log(1 / (1 + exp(-value(i)))); 
                end
            end
        end
        
        % Given a value, evaluates that value at the derivative of the
        % function passed in as well.
        function result = evaluateJacob(obj, value, type)
            % Example for the sigmoid function. Must have an
            % if/equation for each type of transfer function.
            result = zeros(length(value), 1);
            for i = 1:length(value)
                if (type == obj.SIG)
                    y_x = evaluateFunc(obj, value(i), type);
                    result(i) = (1 - y_x) * y_x;
                end
                if (type == obj.LOGSIG)
                    result(i) = 1 / (exp(value(i)) + 1);
                end
            end
            result = diag(result); 
        end    
    end
end










        
        
        
        
        
        