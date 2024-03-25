classdef trefineICNNLowerBoundsOverNDGrid < matlab.unittest.TestCase
    %   Copyright 2024 The MathWorks, Inc.

    methods (TestClassSetup)
        function fixRandomNumberGenerator(testCase)
            % Fix random seed for reproducibility
            seed = rng(0);

            % Add teardown to reset seed to the previous state
            testCase.addTeardown(@() rng(seed))
        end
    end

    methods(Test)       
        function verifyNoNaNsInMinAfterRefinement(testCase)
            % Specify the input size and output size for the network
            inputSize = 1;
            outputSize = 1;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", inputSize, [10 outputSize]);

            % Create hypercubic grid
            granularity = 9;
            intervalSet = linspace(-1,1,granularity);
            V = cell(inputSize,1);
            [V{:}] = ndgrid(intervalSet);

            % Create a dummy netMin full of NaN values
            netMin = {nan([granularity-1 1])};

            % Refine netMin
            refinedNetMin = conslearn.convex.refineICNNLowerBoundsOverNDGrid(net, V, netMin);

            % Verify that there are no NaNs
            testCase.verifyEqual(sum(isnan(refinedNetMin{:}), "all"), 0);
        end

        function verifyNonNanValuesAreUntouchedAfterRefinement(testCase)
            % Specify the input size and output size for the network
            inputSize = 1;
            outputSize = 1;

            % Build fully convex neural network
            net = buildConstrainedNetwork("fully-convex", inputSize, [10 outputSize]);

            % Create hypercubic grid
            granularity = 9;
            intervalSet = linspace(-1,1,granularity);
            V = cell(inputSize,1);
            [V{:}] = ndgrid(intervalSet);

            % Create a dummy netMin with two NaN values
            netMin = {randn([granularity-1 1])};
            nanIdx = [2 7];
            netMin{:}(nanIdx) = nan;

            % Refine netMin
            refinedNetMin = conslearn.convex.refineICNNLowerBoundsOverNDGrid(net, V, netMin);

            % Verify that there are no NaNs
            testCase.verifyEqual(sum(isnan(refinedNetMin{:}), "all"), 0);

            % Verify that non-NaNs are untouched after refinement
            nonNanIdx = setdiff(1:length(netMin{:}), nanIdx);
            testCase.verifyEqual(refinedNetMin{:}(nonNanIdx), netMin{:}(nonNanIdx));
        end
    end
    
end