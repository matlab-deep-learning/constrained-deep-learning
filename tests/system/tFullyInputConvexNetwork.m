classdef tFullyInputConvexNetwork < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        PndActivationFunctionSet = {"softplus", "relu"};
        TargetGeneratorFunctionSet = iCreateTargetGeneratorFunctions;
    end

    methods (Test)
        function verifyNetworkOutputIsFullyConvex(testCase, PndActivationFunctionSet, TargetGeneratorFunctionSet)
            % Create training data
            xTrain = -2:0.01:2;
            tTrain = TargetGeneratorFunctionSet(xTrain);

            % Package dataset into a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain), ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"], ...
                OutputEnvironment="cpu");

            % Build fully convex network
            inputSize = 1;
            numHiddenUnits = [16 8 4 1];
            ficnn = buildConstrainedNetwork("fully-convex",inputSize,numHiddenUnits, ...
                ConvexNonDecreasingActivation=PndActivationFunctionSet);

            % Train fully convex network. Use just 1 epoch.
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.01;
            ficnn = trainConstrainedNetwork("fully-convex", ficnn, mbqTrain, ...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs, ...
                InitialLearnRate=initialLearnRate, ...
                Decay=decay, ...
                LossMetric="mae");

            % Check convexity. Take points A = (a, f(a)) and B = (b, f(b)), 
            % where a<b, and f(x) is the network output for an input x, and
            % verify that point C = (c, f(c)), where c is in range (a, b), 
            % lies below the straight line connecting points A and B
            a = 1;
            b = 3;
            f_a = predict(ficnn, a);
            f_b = predict(ficnn, b);
            m = (f_b - f_a)/(b - a); 
            y0 = f_a - m*a;
            
            % Apply an admissibility constant, epsilon, to test for
            % convexity up to precision errors
            epsilon = 1e-5;

            % Verify that f(c) < (m*c + y0);
            dx = 0.1;
            for c = a+dx:dx:b-dx
                f_c = predict(ficnn, c);
                testCase.verifyLessThanOrEqual(f_c, m*c + y0 + epsilon);
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

end