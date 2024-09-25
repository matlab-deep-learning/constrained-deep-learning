classdef tPartiallyInputConvexNetwork < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        PndActivationFunctionSet = {"softplus", "relu"};
        ActivationFunctionSet = {"relu", "tanh", "fullsort"};
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
        function verifyNetworkIsPartiallyConvex(testCase, PndActivationFunctionSet, ActivationFunctionSet, TargetGeneratorFunctionSet)
            % Create training data for a network with two channels by
            % adding random noise to a partially convex function 
            % f(x1, x2) = x1^3 + x2^4. This function is convex in x2.
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
            picnn = buildConstrainedNetwork("partially-convex",inputSize,numHiddenUnits,...
                ConvexNonDecreasingActivation=PndActivationFunctionSet,...
                Activation=ActivationFunctionSet,...
                ConvexChannelIdx=2);

            % Train the partially convex network
            maxEpochs = 1;
            initialLearnRate = 0.05;
            decay = 0.05;
            lossMetric = "mae";
            picnn = trainConstrainedNetwork("partially-convex",picnn,mbqTrain,...
                TrainingMonitor=false, ...
                MaxEpochs=maxEpochs,...
                InitialLearnRate=initialLearnRate,...
                Decay=decay,...
                LossMetric=lossMetric);

            % Check convexity in x2. Take points A = (x1, a, f(x1, a)) and
            % B = (x1, b, f(x1, b)), where x1 is a constant, a<b, and
            % f(x1,x2) is the network output for an input (x1, x2). Verify
            % that point C = (x1, c, f(c)), where c is in the range (a, b),
            % lies below the straight line connecting points A and B.
            x1 = 2;
            a = 1;
            b = 3;
            f_a = predict(picnn, [x1, a]);
            f_b = predict(picnn, [x1, b]);
            m = (f_b - f_a)/(b - a);
            y0 = f_a - m*a;

            % Apply an admissibility constant, epsilon, to test for
            % convexity up to precision errors
            epsilon = 1e-5;

            % Verify that f(c) < (m*c + y0)
            dx = 0.1;
            for c = a+dx:dx:b-dx
                f_c = predict(picnn, [x1, c]);
                testCase.verifyLessThanOrEqual(f_c, m*c + y0 + epsilon);
            end
        end
    end
end

function param = iCreateTargetGeneratorFunctions()
% Function handles to generate tTrain using x1 and x2

% t = x1^2 - 2*x2 + AWGN
param.ConvexFunctionInX1OnlyWithGaussianNoise = @(x1, x2) x1.^2 - 2*x2 + util.gaussianNoise(x1);

% t = x1^2 + x^2 + AWGN
param.ConvexFunctionInX1AndX2WithGaussianNoise = @(x1, x2) x1.^2 + x2.^2 + util.gaussianNoise(x1);

% t = 2*x1 + x^2 + AWGN
param.ConvexFunctionInX2OnlyWithGaussianNoise = @(x1, x2) 2*x1 + x2.^2 + util.gaussianNoise(x1);

% t = 2*x1 + x^3 + AWGN
param.NonConvexFunctionWithGaussianNoise = @(x1, x2) 2*x1 + x2.^3 + util.gaussianNoise(x1);

% t = 2*x1 + x^2 + sinusoidal noise
param.ConvexFunctionInX2OnlyWithSinusoidalNoise = @(x1, x2) 2*x1 + x2.^2 + util.sinusoidalNoise(x2);
end