classdef NeuralNet
    % NeuralNet class is a container for all of the hyper
    % paramaters that a user would want to tweak in a neural net
    % and not worry about the underlying weights, biases, and 
    % algorithms that are used to tweak these paramaters. 
    
    properties (Access=private)
        w %  The Cell object that holds all of the weight matrices
          %  for each layer of the neural network.
        b %  The Cell object that stores all of the vectors for each
          %  layer of the neural network.
        layers % Number of layers in the network.
        transFuncs % A record of all of the transfer functions.
        learning % Type of learning for this network.
    end
    

    
    % Define transfer function strings, and learning strings.
    properties (Constant)
        % For every transfer function we need to code a private function
        % for its actual value and its derivation.
        
        SIG = 0; PURELIN = 1; 
        TRANS_FUNC_LOW_BOUND = 0; TRANS_FUNC_HIGH_BOUND = 1;
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
        function obj = NeuralNet(dimensions, transFuncs, learning)
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
            obj.w = {};
            obj.w{1} = zeros(dimensions(2), dimensions(1));
            obj.b = {}; 
            obj.b{1} = zeros(dimensions(1), 1);
            for i = 3:layers
               obj.w{i - 1} = zeros(dimensions(i), dimensions(i -1));
               obj.b{i - 1} = zeros(dimensions(i), 1);
            end
            
            % Store the transformation functions.
            obj.transFuncs = transFuncs;
            
            % Store the learning type.
            obj.learning = learning;
            
            % Store the number of layers
            obj.layers = layers;
        end
        
        % Takes in a set of input vectors that have the same number of
        % elements as the first layer of the network. Takes in a set of t
        % vectors that match the number of elements of the output layer.
        % Also the number of p vectors and number of t vectors must be the
        % same. Trains the network with forward propogation and back
        % propogation. Updates the weights according to the learning type.
        function train(obj, p, t)

        end
        
        % Takes in a set of input vectors that must match the number of
        % elements in the input layer of the network. Returns a set of
        % vectors that represent the actual values produced from the test
        % inputs.
        function actual = test(obj, p)
            
        end
        
        % Takes in a cell matrix equivelant that is equivelant dimensions
        % of the dimensions passed in through the constructor. Sets this
        % objects weight to the weights passed in.
        function initWeights(obj, weights)
            
        end
    end
    
    methods (Access=private)
        
        % Not sure what these functions need just yet. They do need to
        % store values differently for different types of learning.
        function result = forwardPropogation()
            
        end
        
        function result = backwardPropogation()
            
        end
        
        function result = updateWeights()
            
        end
        
        % Given a value evaluates that value at the type of function
        % also passed through.
        function result = evaluateFunc(obj, value, type)
            % Example for the sigmoid function. Must have
            % an if/equation for each type of transfer function.
            if (type == obj.SIG)
               result = 1 / (1 + exp(-value));
            end
        end
        
        % Given a value, evaluates that value at the derivative of the
        % function passed in as well.
        function result = evaluateDeriv(obj, value, type)
            % Example for the sigmoid function. Must have an
            % if/equation for each type of transfer function.
            if (type == obj.SIG)
                y_x = evaluateFunc(obj, value, type);
                result = (1 - y_x) * y_x;
            end
        end    
    end
end

