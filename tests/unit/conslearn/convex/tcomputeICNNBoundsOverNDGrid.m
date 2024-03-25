classdef tcomputeICNNBoundsOverNDGrid < matlab.unittest.TestCase
    %   Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        InputSizeSet = {1, 2, 3};
    end

    methods (TestClassSetup)
        function fixRandomNumberGenerator(testCase)
            % Fix random seed for reproducibility
            seed = rng(0);

            % Add teardown to reset seed to the previous state
            testCase.addTeardown(@() rng(seed))
        end
    end

    methods(Test)
        function verifyOutputsHaveTheCorrectShape(testCase)
            % Specify the input size and output size for the network
            inputSize = 1;
            outputSize = 3;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", inputSize, [10 outputSize]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(inputSize,1);
            [V{:}] = ndgrid(intervalSet);

            % Compute ICNN bounds over ND grid
            [netMin, netMax, netPred] = conslearn.convex.computeICNNBoundsOverNDGrid(net, V, outputSize);

            % Verify shape of netPred is correct
            for i = 1:outputSize
                testCase.verifySize(netMin{i}, size(V{1}) - [1 0]);
                testCase.verifySize(netMax{i}, size(V{1}) - [1 0]);
                testCase.verifySize(netPred{i}, size(V{1}));
            end
        end

        function verifyNetPredIsCorrectFor1D(testCase)
            % Specify the input size and output size for the network
            inputSize = 1;
            outputSize = 3;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", inputSize, [10 outputSize]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(inputSize,1);
            [V{:}] = ndgrid(intervalSet);

            % Compute ICNN bounds over ND grid to compute netPred
            [~,~,netPred] = conslearn.convex.computeICNNBoundsOverNDGrid(net, V, outputSize);

            % Get the network's actual prediction using predict
            Z = predict(net, V{:});

            % Verify that netPred is equal to the network's inference for V
            testCase.verifyEqual(cat(2, netPred{:}), Z);
        end

        function verifyNetMaxIsCorrectFor1D(testCase)
            % Specify the input size and output size for the network
            inputSize = 1;
            outputSize = 1;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", inputSize, [10 outputSize]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(inputSize,1);
            [V{:}] = ndgrid(intervalSet);

            % Compute ICNN bounds over ND grid to compute netPred
            [~, netMax, netPred] = conslearn.convex.computeICNNBoundsOverNDGrid(net, V, outputSize);
            netPred = netPred{:};
            netMax = netMax{:};

            % Manually compare each consecutive pairs of elements in
            % netPred to find the greatest. That should be the element
            % stored in netMax.
            for i = 1:length(netPred)-1
                netMaxExpected = max(netPred(i), netPred(i+1));
                netMaxActual = netMax(i);
                testCase.verifyEqual(netMaxActual, netMaxExpected);
            end
        end
    end
end