classdef tFullyInputMonotonicNetwork < matlab.unittest.TestCase

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
        function verifyNetworkOutputsIsFullyMonotonicIncreasing(testCase, ActivationFunctionSet, pNormSet, TargetGeneratorFunctionSet)
            % Create training data
            xTrain = -2:0.01:2;
            tTrain = TargetGeneratorFunctionSet(xTrain);

            % Create a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'.
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain), ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"],...
                OutputEnvironment="cpu");

            % Create a fully monotonic increasing network
            inputSize = 1;
            numHiddenUnits = [16 8 4 1];
            pNorm = pNormSet;
            fimnn = buildConstrainedNetwork("fully-monotonic", inputSize, numHiddenUnits, ...
                Activation=ActivationFunctionSet, ...
                MonotonicTrend="increasing",...
                pNorm=pNorm);

            % Train the fully monotonic network
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.05;
            lossMetric = "mae";
            fimnn = trainConstrainedNetwork("fully-monotonic", fimnn, mbqTrain, ...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs, ...
                InitialLearnRate=initialLearnRate, ...
                Decay=decay, ...
                LossMetric=lossMetric,...
                pNorm=pNorm);

            % Check monotonicity of output. Iteratively take points A = (a,
            % f(a) and B = (a+dx, f(a+dx)) and check that f(a+dx) > f(a).
            dx = 0.1;

            % Apply an admissibility constant, epsilon, to test for
            % monotonicity up to precision errors
            epsilon = 1e-5;
            
            for a = -5:dx:5
                f_a = predict(fimnn, a);
                f_a_dx = predict(fimnn, a+dx);
                testCase.verifyGreaterThanOrEqual(f_a_dx, f_a - epsilon);
            end
        end

        function verifyNetworkOutputsIsFullyMonotonicDecreasing(testCase, ActivationFunctionSet, pNormSet, TargetGeneratorFunctionSet)
            % Create training data
            xTrain = -2:0.01:2;
            tTrain = TargetGeneratorFunctionSet(xTrain);

            % Create a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'.
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain), ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"]);

            % Create a fully monotonic increasing network
            inputSize = 1;
            numHiddenUnits = [16 8 4 1];
            pNorm = pNormSet;
            fimnn = buildConstrainedNetwork("fully-monotonic", inputSize, numHiddenUnits, ...
                Activation=ActivationFunctionSet, ...
                MonotonicTrend="decreasing",...
                pNorm=pNorm);

            % Train the fully monotonic network
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.05;
            lossMetric = "mae";
            fimnn = trainConstrainedNetwork("fully-monotonic", fimnn, mbqTrain, ...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs, ...
                InitialLearnRate=initialLearnRate, ...
                Decay=decay, ...
                LossMetric=lossMetric,...
                pNorm=pNorm);

            % Check monotonicity of output. Iteratively take points A = (a,
            % f(a) and B = (a+dx, f(a+dx)) and check that f(a+dx) < f(a).
            dx = 0.1;

            % Apply an admissibility constant, epsilon, to test for
            % monotonicity up to precision errors
            epsilon = 1e-5;

            for c = -5:dx:5
                f_a = predict(fimnn, c);
                f_a_dx = predict(fimnn, c+dx);
                testCase.verifyLessThanOrEqual(f_a_dx, f_a + epsilon);
            end
        end
    end

end

function param = iCreateTargetGeneratorFunctions()
% Function handles to generate tTrain using xTrain

% t = x + AWGN
param.MonotonicIncreasing = @(x) x + util.gaussianNoise(x);

% t = -x + AWGN
param.MonotonicDecreasing = @(x) -x + util.gaussianNoise(x);

% t = x^2 + AWGN
param.NonMonotonic = @(x) x.^2 + util.gaussianNoise(x);
end