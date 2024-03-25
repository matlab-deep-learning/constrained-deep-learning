classdef tgetLipschitzParameterIdx < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.
    
    methods(TestClassSetup)
        % Shared setup for the entire test class
    end
    
    methods(TestMethodSetup)
        % Setup for each test
    end
    
    methods(Test)
        % Test methods
        
        function verifyWorksForNetworkWithOneFullyConnectedLayer(testCase)
            % Create a simple network with one fully connected layer
            net = dlnetwork([ ...
                featureInputLayer(3), ...
                reluLayer, ...
                fullyConnectedLayer(10)]);
            
            % Define Lipschitz upper bound
            lipschitzConstant = 4;

            % Call conslearn.lipschitz.getLipschitzParameterIdx to get the
            % indices of learnables that correspond to weights with
            % Lipschitz constraints and their Lipschitz upper bounds
            [lipschitzParams, lipschitzIdx] = conslearn.lipschitz.getLipschitzParameterIdx(net, lipschitzConstant);

            % Test that the lipschitzIdx is [1 0]', since the param in the
            % net.Learnables table of the network are weights which should
            % be Lipschitz constrained and the second are biases which
            % should not be Lipschitz constrained
            testCase.verifyEqual(lipschitzIdx, [true false]');

            % Test that the value of the lipschitzParams table is equal to
            % the lipschitz constant specified
            actualLipschitzBounds = lipschitzParams.Value(lipschitzIdx);
            testCase.verifyEqual(extractdata(actualLipschitzBounds{:}), lipschitzConstant);
        end

        function verifyWorksForNetworkWithMultipleFullyConnectedLayers(testCase)
            % Create a simple network with one fully connected layer
            net = dlnetwork([ ...
                featureInputLayer(3), ...
                reluLayer, ...
                fullyConnectedLayer(10), ...
                reluLayer, ...
                fullyConnectedLayer(10), ...
                reluLayer, ...
                fullyConnectedLayer(10)]);

            % Define Lipschitz constant input
            lipschitzConstant = 10;

            % Call conslearn.lipschitz.getLipschitzParameterIdx to get the
            % indices of learnables that correspond to weights with
            % Lipschitz constraints and their Lipschitz upper bounds
            [lipschitzParams, lipschitzIdx] = conslearn.lipschitz.getLipschitzParameterIdx(net, lipschitzConstant);

                % Test that the lipschitzIdx is [1 0]', since the param in the
            % net.Learnables table of the network are weights which should
            % be Lipschitz constrained and the second are biases which
            % should not be Lipschitz constrained
            testCase.verifyEqual(lipschitzIdx, repmat([true false]', 3, 1));

            % Test that the value of the lipschitzParams table is equal to
            % the expected lipschitz constant
            expectedLipschitzBounds = repmat(lipschitzConstant^(1/3), 3, 1);
            actualLipschitzBounds = lipschitzParams.Value(lipschitzIdx);
            testCase.verifyEqual(cellfun(@extractdata, actualLipschitzBounds), expectedLipschitzBounds);
        end
    end
    
end