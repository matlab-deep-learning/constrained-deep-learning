classdef tconvexNetworkOutputBounds < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        NumInputSet = {1, 2, 3};
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
        function verifyOutputsHaveCorrectShape(testCase, NumInputSet)
            % Get the input size for the network
            numInputs = NumInputSet;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", numInputs, [10 3]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(numInputs,1);
            [V{:}] = ndgrid(intervalSet);

            % Calculate network output bounds
            [netMin,netMax,netPred] = convexNetworkOutputBounds(net,V);

            % Size dimensions to check
            checkDims = 1:numInputs;

            % Verify that the size of netPred is equal to the size of V
            testCase.verifyEqual(size(netPred{1}, checkDims), size(V{1}, checkDims));

            % Verify that the size of netMax and netMin are equal to the
            % size of V minus 1
            testCase.verifyEqual(size(netMax{1}, checkDims), size(V{1}, checkDims) - 1);
            testCase.verifyEqual(size(netMin{1}, checkDims), size(V{1}, checkDims) - 1);
        end

        function verifyNetMinHasNoNansForRefineLowerBoundsTrue(testCase, NumInputSet)
            % Get the input size for the network
            numInputs = NumInputSet;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", numInputs, [10 1]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(numInputs,1);
            [V{:}] = ndgrid(intervalSet);

            % Calculate network output bounds
            netMin = convexNetworkOutputBounds(net,V, RefineLowerBounds=true);

            % Verify there are no NaNs in netMin
            testCase.verifyEqual(sum(isnan(netMin{:}), "all"), 0)
        end

        function verifyNetMaxIsGreaterThanNetMin(testCase, NumInputSet)
            % Get the input size for the network
            numInputs = NumInputSet;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", numInputs, [10 1]);

            % Create hypercubic grid
            intervalSet = linspace(-1,1,9);
            V = cell(numInputs,1);
            [V{:}] = ndgrid(intervalSet);

            % Calculate network output bounds
            [netMin,netMax] = convexNetworkOutputBounds(net,V);

            % Verify that corresponding elements of netMax are always
            % greater than netMin
            testCase.verifyGreaterThan(netMax{:}, netMin{:});
        end

        function verifyNetMaxAndNetMinAreSatisfied(testCase)
            % Specify numInputs = 1, and numOutputs = 1
            numInputs = 1;
            numOutputs = 1;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", numInputs, [10 numOutputs]);

            % Specify a region to check from 0 to 1.
            xMin = 0;
            xMax = 1;
            V = {[xMin xMax]};

            % Calculate networkOutputBounds
            [netMin,netMax] = convexNetworkOutputBounds(net,V);

            % Add single precision error
            netMax{:} = netMax{:} + 1e-6;
            netMin{:} = netMin{:} - 1e-6;

            % Get a vector x to pass throught the network
            for x = linspace(xMin, xMax, 50)
                testCase.verifyGreaterThanOrEqual(predict(net, x), netMin{:}, mat2str(x));
                testCase.verifyLessThanOrEqual(predict(net, x), netMax{:}, mat2str(x));
            end
        end

        function verifyErrorFreeSingleHypercube(testCase, NumInputSet)
            numInputs = NumInputSet;
            numOutputs = 1;
            V = cell(numInputs,1);
            % Single hypercube
            [V{:}] = ndgrid([-1 1]);

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", numInputs, [10 numOutputs]);

            % Calculate networkOutputBounds
            fcn = @()convexNetworkOutputBounds(net,V);

            % Verify no errors
            testCase.verifyWarningFree(fcn);
        end
    end

end