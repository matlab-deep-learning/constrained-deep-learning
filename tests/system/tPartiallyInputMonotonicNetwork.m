classdef tPartiallyInputMonotonicNetwork < matlab.unittest.TestCase

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
        function verifyNetworkIsPartiallyMonotonicIncreasing(testCase, ActivationFunctionSet, pNormSet, TargetGeneratorFunctionSet)
            % Create training data for a network with two channels by
            % adding noise to a partially monotonic function 
            % f(x1, x2) = x1^2 + 2*x2. This is monotonic in x2, but not in
            % x1.
            numSamples = 1024;
            [x1Train,x2Train] = meshgrid(linspace(-1,1,round(sqrt(numSamples))));
            xTrain = [x1Train(:),x2Train(:)];
            tTrain = TargetGeneratorFunctionSet(xTrain(:, 1), xTrain(:, 2));

                % Create a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'.
            xds = arrayDatastore(xTrain);
            tds = arrayDatastore(tTrain);
            cds = combine(xds,tds);
            mbqTrain = minibatchqueue(cds,2,...
                "MiniBatchSize",numSamples,...
                "OutputAsDlarray",[1 1],...
                "OutputEnvironment","cpu",...
                "MiniBatchFormat",["BC","BC"]);

            % Create a partially convex network which is convex in the
            % second channel only.
            inputSize = 2;
            numHiddenUnits = [32 8 1];
            pNorm = pNormSet;
            pimnn = buildConstrainedNetwork("partially-monotonic",inputSize,numHiddenUnits,...
                Activation=ActivationFunctionSet, ...
                MonotonicChannelIdx=2, ...
                MonotonicTrend="increasing",...
                pNorm=pNorm);

            % Train the partially convex network
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.05;
            lossMetric = "mae";
            pimnn = trainConstrainedNetwork("partially-monotonic",pimnn,mbqTrain,...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs,...
                InitialLearnRate=initialLearnRate,...
                Decay=decay,...
                LossMetric=lossMetric,...
                pNorm=pNorm);

            % Verify monotonicity in x2. Iteratively take points 
            % A = (x1, a, f(a)) and B = (x1, a+dx, f(a+dx)) and check that 
            % f(a+dx) > f(a).
            dx = 0.1;
            x1 = 3;

            % Apply an admissibility constant, epsilon, to test for
            % monotonicity up to precision errors
            epsilon = 1e-5;

            for a = -5:dx:5
                f_a = predict(pimnn, [x1, a]);
                f_a_dx = predict(pimnn, [x1, a+dx]);
                testCase.verifyGreaterThanOrEqual(f_a_dx, f_a - epsilon);
            end
        end

        function verifyNetworkIsPartiallyMonotonicDecreasing(testCase, ActivationFunctionSet, pNormSet, TargetGeneratorFunctionSet)
            % Create training data for a network with two channels by
            % adding noise to a partially monotonic function
            % f(x1, x2) = x1^2 + 2*x2. This is monotonic in x2, but not in
            % x1.
            numSamples = 1024;
            [x1Train,x2Train] = meshgrid(linspace(-1,1,round(sqrt(numSamples))));
            xTrain = [x1Train(:),x2Train(:)];
            tTrain = TargetGeneratorFunctionSet(xTrain(:, 1), xTrain(:, 2));

            % Create a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'.
            xds = arrayDatastore(xTrain);
            tds = arrayDatastore(tTrain);
            cds = combine(xds,tds);
            mbqTrain = minibatchqueue(cds,2,...
                "MiniBatchSize",numSamples,...
                "OutputAsDlarray",[1 1],...
                "OutputEnvironment","cpu",...
                "MiniBatchFormat",["BC","BC"]);

            % Create a partially convex network which is convex in the
            % second channel only.
            inputSize = 2;
            numHiddenUnits = [32 8 1];
            pNorm = pNormSet;
            pimnn = buildConstrainedNetwork("partially-monotonic",inputSize,numHiddenUnits,...
                Activation=ActivationFunctionSet, ...
                MonotonicChannelIdx=2, ...
                MonotonicTrend="decreasing",...
                pNorm=pNorm);

            % Train the partially convex network
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.05;
            lossMetric = "mae";
            pimnn = trainConstrainedNetwork("partially-monotonic",pimnn,mbqTrain,...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs,...
                InitialLearnRate=initialLearnRate,...
                Decay=decay,...
                LossMetric=lossMetric,...
                pNorm=pNorm);

            % Verify monotonicity in x2. Iteratively take points
            % A = (x1, a, f(a)) and B = (x1, a+dx, f(a+dx)) and check that
            % f(a+dx) > f(a).
            dx = 0.1;
            x1 = 3;

            % Apply an admissibility constant, epsilon, to test for
            % monotonicity up to precision errors
            epsilon = 1e-5;

            for a = -5:dx:5
                f_a = predict(pimnn, [x1, a]);
                f_a_dx = predict(pimnn, [x1, a+dx]);
                testCase.verifyLessThanOrEqual(f_a_dx, f_a + epsilon);
            end
        end
    end
end

function param = iCreateTargetGeneratorFunctions()
% Function handles to generate tTrain using x1 and x2

% t = x1^2 + 2*x2 + AWGN
param.MonotonicIncreasingInX2OnlyWithGaussianNoise = @(x1, x2) x1.^2 + 2*x2 +  util.gaussianNoise(x1);

% t = x1^2 - 2*x2 + AWGN
param.MonotonicDecreasingInX2OnlyWithGaussianNoise = @(x1, x2) x1.^2 - 2*x2 +  util.gaussianNoise(x1);

% t = x1^2 + 2*x2^2 + AWGN
param.NonMonotonic = @(x1, x2) x1.^2 + 2*x2 +  util.gaussianNoise(x1);

% t = x1 + x2 + AWGN
param.MonotonicIncreasingInX1AndX2 = @(x1, x2) x1 + x2 +  util.gaussianNoise(x1);

% t = x1^4 + x2 + sinusoidal noise
param.MonotonicIcreasingInX2OnlyWithSinusoidalNoise = @(x1, x2) x1.^4 + x2 + util.sinusoidalNoise(x2);

% t = x1^4 + x2 + sinusoidal noise
param.MonotonicDecreasingInX2OnlyWithSinusoidalNoise = @(x1, x2) x1.^4 - x2 + util.sinusoidalNoise(x2);
end