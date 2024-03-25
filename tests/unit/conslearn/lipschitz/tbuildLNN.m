classdef tbuildLNN < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    methods (TestClassSetup)
        function fixRandomNumberGenerator(testCase)
            % Fix random seed for reproducibility
            seed = rng(0);

            % Add teardown to reset seed to the previous state
            testCase.addTeardown(@() rng(seed))
        end
    end

    properties(TestParameter)
        InputSizeSet = struct( ...
            GrayscaleImage = struct( ...
            Input = [28, 28, 1], ...
            ExpectedLayer = ?nnet.cnn.layer.ImageInputLayer), ...
            TricolourImage = struct( ...
            Input = [28, 28, 3], ...
            ExpectedLayer = ?nnet.cnn.layer.ImageInputLayer), ...
            Features = struct( ...
            Input = 10, ...
            ExpectedLayer = ?nnet.cnn.layer.FeatureInputLayer))

        FullyConnectedLayerSizesSet = struct( ...
            OneFCLayer = 8, ...
            TwoFCLayersSameSize = [10, 10], ...
            FourFCLayersSameSize = [10, 10, 10, 10], ...
            TwoFCLayersDifferentSize = [10, 100], ...
            FourFCLayersDifferentSize = [10, 50, 1000, 10])

        ActivationFunctionSet = struct( ...
            FullSort = struct( ...
            Input = 'fullsort', ...
            ExpectedLayer = ?conslearn.layer.FullSortLayer), ...
            ReLU = struct( ...
            Input = 'relu', ...
            ExpectedLayer = ?nnet.cnn.layer.ReLULayer), ...
            Tanh = struct( ...
            Input = 'tanh', ...
            ExpectedLayer = ?nnet.cnn.layer.TanhLayer));

        PNormSet = {1, 2, Inf};

        UpperBoundLipschitzConstantSet = {0.1, 0.5, 1, 2, 5, 10, 100};
    end


    methods(Test)
        function verifyInputLayerIsCorrect(testCase, InputSizeSet)
            import matlab.unittest.constraints.IsOfClass

            % Build Lipschitz upper bound network
            net = conslearn.lipschitz.buildLNN(InputSizeSet.Input, 10);

            % Test that the network has the expected input layer
            testCase.verifyThat(net.Layers(1), IsOfClass(InputSizeSet.ExpectedLayer));
        end

        function verifyLayersAreCorrectForFeatureInput(testCase, FullyConnectedLayerSizesSet, ActivationFunctionSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass
            import matlab.unittest.constraints.IsEqualTo

            % Build Lipschitz upper bound network
            net = conslearn.lipschitz.buildLNN(20, FullyConnectedLayerSizesSet, ...
                Activation=ActivationFunctionSet.Input);

            % Test that the network has the correct number of layers
            % (1 input, N activation and N fullyconnect,
            % where N is the length of the numHiddenUnits input)
            N = length(FullyConnectedLayerSizesSet);
            testCase.verifyThat(net.Layers, HasLength(2*N + 1));

            % Get indices for hidden fully connected layers
            fcLayerIdx = iFindLayerIdxWithType(net, 'nnet.cnn.layer.FullyConnectedLayer');

            % Test that the hidden inputs are assigned correctly to the
            % fully connected layers
            testCase.verifyThat([net.Layers(fcLayerIdx).OutputSize], ...
                IsEqualTo(FullyConnectedLayerSizesSet));

            % Test that the activation layers have the correct class
            expectedActivationLayerIdx = fcLayerIdx - 1;
            testCase.verifyThat(net.Layers(expectedActivationLayerIdx), ...
                IsOfClass(ActivationFunctionSet.ExpectedLayer))
        end

        function verifyLayersAreCorrectForImageInput(testCase, FullyConnectedLayerSizesSet, ActivationFunctionSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass
            import matlab.unittest.constraints.IsEqualTo

            % Build Lipschitz upper bound network
            net = conslearn.lipschitz.buildLNN([20, 20, 3], FullyConnectedLayerSizesSet, ...
                Activation=ActivationFunctionSet.Input);

            % Test that the network has the correct number of layers
            % (1 input, N activation and N fullyconnect,
            % where N is the length of the numHiddenUnits input)
            N = length(FullyConnectedLayerSizesSet);
            testCase.verifyThat(net.Layers, HasLength(2*N + 2));

            % Get indices for flatten layer
            actualFlattenLayerIdx = iFindLayerIdxWithType(net, 'nnet.cnn.layer.FlattenLayer');

            % Test that the flatten layer is the second layer after the
            % image input layer
            testCase.verifyThat(actualFlattenLayerIdx, IsEqualTo(2));

            % Get indices for hidden fully connected layers
            actualFcLayerIdx = iFindLayerIdxWithType(net, 'nnet.cnn.layer.FullyConnectedLayer');

            % Test that the hidden inputs are assigned correctly to the
            % fully connected layers
            testCase.verifyThat([net.Layers(actualFcLayerIdx).OutputSize], ...
                IsEqualTo(FullyConnectedLayerSizesSet));

            % Test that the activation layers have the correct class
            expectedActivationLayerIdx = actualFcLayerIdx - 1;
            testCase.verifyThat(net.Layers(expectedActivationLayerIdx), ...
                IsOfClass(ActivationFunctionSet.ExpectedLayer))
        end

        function verifyNetworkHasACorrectLipschitzUpperBound(testCase, PNormSet, UpperBoundLipschitzConstantSet)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.RelativeTolerance
            import matlab.unittest.constraints.IsLessThan

            % Build Lipschitz upper bound network
            net = conslearn.lipschitz.buildLNN(10, [10, 50, 1000, 10], ...
                UpperBoundLipschitzConstant=UpperBoundLipschitzConstantSet, ...
                pNorm=PNormSet);

            % Calculate Lipschitz upper bound
            actualLipschitzUpperBound = lipschitzUpperBound(net, PNormSet);
            expectedLipschitzUpperBound = UpperBoundLipschitzConstantSet;

            % Test that the calculated Lipschitz upper bound is less than
            % the expected Lipschitz upper bound or equal to the expected
            % Lipschitz upper bound within a relative tolerance of 1e-5
            % which accounts for precision errors
            testCase.verifyThat(extractdata(double(actualLipschitzUpperBound)), ...
                IsLessThan(expectedLipschitzUpperBound) | ...
                IsEqualTo(expectedLipschitzUpperBound, Within=RelativeTolerance(1e-5)));
        end
    end
end

function idx = iFindLayerIdxWithType(net, type)
layerClasses = arrayfun(@class, net.Layers, 'UniformOutput', false);
idx = find(cellfun(@(x) strcmp(x, type), layerClasses));
end