classdef tFullyInputConvexNetwork < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        PndActivationFunctionSet = {"softplus", "relu"};
        TargetGeneratorFunctionSet = iCreateTargetGeneratorFunctions;
    end

    methods (Test)
        function verifyNetworkOutputIsFullyConvexMLP(testCase, PndActivationFunctionSet, TargetGeneratorFunctionSet)
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

        function verifyNetworkOutputIsFullyConvexForFeatureInputCNN(testCase, TargetGeneratorFunctionSet)
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
                MiniBatchFormat=["BTC", "BC"], ...
                OutputEnvironment="cpu");

            % Build fully convex network
            inputSize = 1;
            outputSize = 1;
            filterSize = 2;
            numFilters = 8;
            ficnn = buildConvexCNN( ...
                inputSize, ...
                outputSize, ...
                filterSize, ...
                numFilters);

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

        function verifyNetworkOutputIsFullyConvexForImageInputCNN(testCase)
            % Get training data. Fix random seed for reproducibility.
            rng(0, "twister");
            inputSize = [28 28 3];
            outputSize = 1;
            batchSize = 10;
            xTrain = randn([inputSize batchSize]);
            tTrain = randn([outputSize batchSize]);

            % Package dataset into a minibatchqueue that can be used by
            % 'trainConstrainedNetwork'
            xds = arrayDatastore(xTrain, IterationDimension=4);
            tds = arrayDatastore(tTrain, IterationDimension=2);
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=size(xTrain, 4), ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["SSCB", "BC"], ...
                OutputEnvironment="cpu");

            % Build fully convex network
            filterSize = 2;
            numFilters = 8;
            ficnn = buildConvexCNN( ...
                inputSize, ...
                outputSize, ...
                filterSize, ...
                numFilters);

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

            % Check convexity. Take at random points x1 and x2, where x1 is
            % not equal to x2 and verify that the following inequality is true:
            %
            % f((1-lambda)*x1 + lambda*x2) <= (1-lambda)*f(x1) + lambda*f(x2)
            %
            % Where:
            %   f(x): the network output for an input x, 
            %   lambda: a scalar constant in the range [0, 1].

            % Create random points x1 and x2. Assume that they are not equal.
            rng(0, "twister");
            x1 = randn(inputSize);
            x2 = randn(inputSize);
            
            % Apply an admissibility constant, epsilon, to test for
            % convexity up to precision errors
            epsilon = 1e-7;

            % Verify convexity by taking 9 equidistant samples for lambda 
            % in the range [0, 1] 
            for lambda = linspace(0, 1, 9)
                lhs = predict(ficnn, (1-lambda) * x1 + lambda * x2);
                rhs = (1-lambda)*predict(ficnn, x1) + lambda*predict(ficnn, x2);
                testCase.verifyLessThanOrEqual(lhs, rhs + epsilon, ...
                    "Trained network is not convex.");
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