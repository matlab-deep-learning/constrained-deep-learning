classdef tLipschitzContinuousNetwork < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ActivationFunctionSet = {"relu", "tanh", "fullsort"};
        pNormSet = {1, 2, Inf};
        TargetGeneratorFunctionSet = iCreateTargetGeneratorFunctions;
    end

    methods (TestClassSetup)
        function fixRandomNumberGenerator(testCase)
            % Fix random seed for reproducibility
            seed = rng(0);

            % Add teardown to reset seed to the previous state
            testCase.addTeardown(@() rng(seed))
        end
    end

    methods (Test)
        function verifyNetworkOutputIsWithinCalculatedLipschitzBound(testCase, ActivationFunctionSet, pNormSet, TargetGeneratorFunctionSet)
            % Create training data by adding a sine wave as noise to a
            % monotonic function such as y = x^3
            numSamples = 512;
            xTrain = -2*rand(numSamples, 1)+1;
            xTrain = sort(xTrain);
            tTrain = TargetGeneratorFunctionSet(xTrain);

            % Create a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'.
            xds = arrayDatastore(xTrain);
            tds = arrayDatastore(tTrain);
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain), ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"]);

            % Create a Lipschitz constrained network
            inputSize = 1;
            numHiddenUnits = [32 16 8 1];
            upperBoundLipschitzConstant = 4;
            pNorm = pNormSet;
            lnn = buildConstrainedNetwork("lipschitz", inputSize, numHiddenUnits, ...
                Activation=ActivationFunctionSet, ...
                UpperBoundLipschitzConstant=upperBoundLipschitzConstant, ...
                pNorm=pNorm);

            % Train the Lipschitz constrained network
            maxEpochs = 1;
            initialLearnRate = 0.5;
            decay = 0.5;
            lossMetric = "mse";
            lnn = trainConstrainedNetwork("lipschitz", lnn, mbqTrain, ...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs,...
                InitialLearnRate=initialLearnRate,...
                Decay=decay,...
                LossMetric=lossMetric);

            % Calculate the actual Lipschitz upper bound of the network
            calculatedLipschitzConstant = lipschitzUpperBound(lnn, pNorm);

            % Verify that the network is lipschitz continuous in the
            % calculated lipschitz constant, by checking that it satisfies
            % the Lipschitz inequality:
            % ||f(x1) - f(x2)||p <= lambda * ||x1 - x2||p + epsilon
            % Where f(x) is the network output given an input x, lambda is
            % the calculated lipschitz constant and ||x||p is the p-norm of
            % x, and epsilon is an admissability constant, to account for
            % precision errors.
            % Iterate through different values in the network and check
            % this is satisfied.
            dx = 0.1;

            % Apply an admissibility constant, epsilon, to test for
            % Lipschitz continuity up to precision errors
            epsilon = 1e-5;

            for x = -2:dx:2
                x1 = x;
                x2 = x+dx;
                f_x1 = predict(lnn, x1);
                f_x2 = predict(lnn, x2);
                pnorm_f_x = norm(f_x1-f_x2, pNorm);
                pnorm_x = norm(x1-x2, pNorm);
                testCase.verifyLessThanOrEqual(pnorm_f_x, calculatedLipschitzConstant * pnorm_x + epsilon);
            end
        end
    end
end

function param = iCreateTargetGeneratorFunctions()
% Function handles to generate tTrain using x

% t = x + AWGN
param.MonotonicIncreasing = @(x) x + util.gaussianNoise(x);

% t = -x + AWGN
param.MonotonicDecreasing = @(x) -x + util.gaussianNoise(x);

% t = x^2 + AWGN
param.ConvexFunctionWithGaussianNoise = @(x) x.^2 + util.gaussianNoise(x);

% t = x^4 + sinusoidal noise
param.ConvexFunctionWithSinusoidalNoise = @(x) x.^4 + util.sinusoidalNoise(x);

% t - x^3 + sinusoidal noise
param.ConcaveFunctionWithSinusoidalNoise =@(x) x.^3 + util.sinusoidalNoise(x);
end