classdef ttrainConstrainedNetwork < matlab.unittest.TestCase
    %   Copyright 2024 The MathWorks, Inc.

    methods (TestClassSetup)
        function fixRandomNumberGenerator(testCase)
            % Fix random seed for reproducibility
            seed = rng(0);

            % Add teardown to reset seed to the previous state
            testCase.addTeardown(@() rng(seed))
        end
    end

    properties(TestParameter)
        ValidConvexConstraintSet = iCreateValidConvexConstraintSet;
        ValidMonotonicConstraintSet = iCreateValidMonotonicConstraintSet;
        ValidPNormSet = {1, 2, inf};
        ValidUpperBoundLipschitzConstantSet = {0.5, 4};
        ValidShuffleMinibatchesSet = {true false};
        ValidMaxEpochsSet = {1, 10};
        ValidInitialLearnRateSet = {0.5, 1, 5};
        ValidDecaySet = {0.01, 1};
        ValidRegressionLossMetricSet = {"mse","mae"};
        ValidClassificationLossMetricSet = {"crossentropy"};
    end

    methods(Test)
        function verifyConvexNetworkCanBeTrainedForFeatureInput(testCase, ValidConvexConstraintSet)
            % Define network building parameters
            constraint = ValidConvexConstraintSet;
            inputSize = 10;
            numHiddenUnits = [512 256 128 10];

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            % Create training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyConvexNetworkCanBeTrainedForImageInput(testCase, ValidConvexConstraintSet)
            % Define network building parameters
            constraint = ValidConvexConstraintSet;
            inputSize = [28 28 3];
            numHiddenUnits = [512 256 128 10];

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            % Create random training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain, IterationDimension=4);
            tds = arrayDatastore(tTrain, IterationDimension=2);
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["SSCB" "CB"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyMonotonicNetworkCanBeTrainedForFeatureInput(testCase, ValidMonotonicConstraintSet, ValidPNormSet)
            % Define network building parameters
            constraint = ValidMonotonicConstraintSet;
            inputSize = 10;
            numHiddenUnits = [512 256 128 10];
            pNorm = ValidPNormSet;

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            % Create training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                pNorm = pNorm);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyMonotonicNetworkCanBeTrainedForImageInput(testCase, ValidMonotonicConstraintSet, ValidPNormSet)
            % Define network building parameters
            constraint = ValidMonotonicConstraintSet;
            inputSize = [28 28 3];
            numHiddenUnits = [512 256 128 10];
            pNorm = ValidPNormSet;

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            % Create random training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain, IterationDimension=4);
            tds = arrayDatastore(tTrain, IterationDimension=2);
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["SSCB" "CB"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                pNorm = pNorm);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyLipschitzNetworkCanBeTrainedForFeatureInput(testCase, ValidPNormSet, ValidUpperBoundLipschitzConstantSet)
            % Define network building parameters
            constraint = "lipschitz";
            pNorm = ValidPNormSet;
            upperBoundLipschitzConstant = ValidUpperBoundLipschitzConstantSet;
            inputSize = 10;
            numHiddenUnits = [512 256 128 10];

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                UpperBoundLipschitzConstant = upperBoundLipschitzConstant, ...
                pNorm = pNorm);

            % Create training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain');
            tds = arrayDatastore(tTrain');
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["BC", "BC"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();
            trainOptions.UpperBoundLipschitzConstant = upperBoundLipschitzConstant;
            trainOptions.pNorm = pNorm;

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                UpperBoundLipschitzConstant = trainOptions.UpperBoundLipschitzConstant, ...
                pNorm = trainOptions.pNorm);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyLipschitzNetworkCanBeTrainedForImageInput(testCase, ValidPNormSet, ValidUpperBoundLipschitzConstantSet)
            % Define network building parameters
            constraint = "lipschitz";
            pNorm = ValidPNormSet;
            upperBoundLipschitzConstant = ValidUpperBoundLipschitzConstantSet;
            inputSize = [28 28 3];
            numHiddenUnits = [512 256 128 10];

            % Build network
            netBefore = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            % Create random training data
            noObservations = 128;
            xTrain = randn([inputSize noObservations]);
            tTrain = randn([numHiddenUnits(end) noObservations]);
            xds = arrayDatastore(xTrain, IterationDimension=4);
            tds = arrayDatastore(tTrain, IterationDimension=2);
            cds = combine(xds, tds);
            mbqTrain = minibatchqueue(cds, 2, ...
                MiniBatchSize=length(xTrain)/2, ...
                OutputAsDlarray=[1 1], ...
                MiniBatchFormat=["SSCB" "CB"]);

            % Define training options
            trainOptions = iDefaultTrainOptions();
            trainOptions.UpperBoundLipschitzConstant = upperBoundLipschitzConstant;
            trainOptions.pNorm = pNorm;

            % Define network training function
            fcn = @() trainConstrainedNetwork(constraint, netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                UpperBoundLipschitzConstant = trainOptions.UpperBoundLipschitzConstant, ...
                pNorm = trainOptions.pNorm);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForValidShuffleMinibatches(testCase, ValidShuffleMinibatchesSet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 1;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                ShuffleMinibatches=ValidShuffleMinibatchesSet);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForValidMaxEpochs(testCase, ValidMaxEpochsSet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 1;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = ValidMaxEpochsSet, ...
                LossMetric = trainOptions.LossMetric);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForValidInitialLearnRate(testCase, ValidInitialLearnRateSet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 1;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                InitialLearnRate=ValidInitialLearnRateSet);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForValidDecay(testCase, ValidDecaySet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 1;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = trainOptions.LossMetric, ...
                Decay=ValidDecaySet);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForRegressionMetric(testCase, ValidRegressionLossMetricSet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 1;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = ValidRegressionLossMetricSet);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyFullyConvexNetworkCanBeTrainedForClassificationMetric(testCase, ValidClassificationLossMetricSet)
            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 10;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = trainOptions.MaxEpochs, ...
                LossMetric = ValidClassificationLossMetricSet);

            % Verify training the network is warning free
            netAfter = testCase.verifyWarningFree(fcn);

            % Verify the network output is a dlnetwork
            testCase.verifyClass(netAfter, "dlnetwork");

            % Verify that the network learnables are now changed
            testCase.verifyNotEqual(netAfter.Learnables, netBefore.Learnables);
        end

        function verifyErrorForMaxEpochsEqualToInf(testCase)
            import matlab.unittest.constraints.Throws

            % Specify network's input size and output size
            inputSize = 10;
            outputSize = 10;

            % Build default fully-convex network
            netBefore = iBuildDefaultFullyConvexNetwork(inputSize, outputSize);

            % Build default minibatchqueue
            mbqTrain = iBuildDefaultMinibatchqueue(inputSize, outputSize);

            % Define training options
            trainOptions = iDefaultTrainOptions();

            % Define network training function
            fcn = @() trainConstrainedNetwork("fully-convex", netBefore, mbqTrain, ...
                TrainingMonitor = trainOptions.TrainingMonitor, ...
                MaxEpochs = inf);

            % Verify that we throw an error and do not allow infinite
            % number of epochs
            testCase.verifyThat(fcn, Throws(?MException));
        end
    end

end

function trainOptions = iDefaultTrainOptions()
trainOptions.TrainingMonitor = false;
trainOptions.MaxEpochs = 1;
trainOptions.LossMetric = "mse";
end

function net = iBuildDefaultFullyConvexNetwork(inputSize, outputSize)
constraint = "fully-convex";
numHiddenUnits = [512 256 128 outputSize];

net = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);
end

function mbq = iBuildDefaultMinibatchqueue(inputSize, outputSize)
noObservations = 128;
xTrain = randn([inputSize noObservations]);
tTrain = randn([outputSize noObservations]);
xds = arrayDatastore(xTrain');
tds = arrayDatastore(tTrain');
cds = combine(xds, tds);
mbq = minibatchqueue(cds, 2, ...
    MiniBatchSize=length(xTrain)/2, ...
    OutputAsDlarray=[1 1], ...
    MiniBatchFormat=["BC", "BC"]);
end

function param = iCreateValidConvexConstraintSet()
param.FullyConvex = "fully-convex";
param.PartiallyConvex = "partially-convex";
end

function param = iCreateValidMonotonicConstraintSet()
param.FullyMonotonic = "fully-monotonic";
param.PartiallyMonotonic = "partially-monotonic";
end