classdef tbuildPMNN < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

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
            ExpectedLayerClass = 'conslearn.layer.FullSortLayer'), ...
            ReLU = struct( ...
            Input = 'relu', ...
            ExpectedLayerClass = 'nnet.cnn.layer.ReLULayer'), ...
            Tanh = struct( ...
            Input = 'tanh', ...
            ExpectedLayerClass = 'nnet.cnn.layer.TanhLayer'));

        MonotonicChannelIdxSet = {1, 1:3, [2 4 6], [5:7 12]};

        MonotonicTrendSet = struct( ...
            Increasing = struct( ...
            Input = 'increasing', ...
            ExpectedResidualScaling = 1), ...
            Decreasing = struct( ...
            Input = 'decreasing', ...
            ExpectedResidualScaling = -1));

        PNormSet = {1, 2, Inf};

        ResidualScalingSet = {0.1, 0.5, 1, 2, 5, 10, 100};
    end

    methods(Test)
        function verifyInputLayerIsCorrect(testCase, InputSizeSet)
            import matlab.unittest.constraints.IsOfClass

            % Build Lipschitz upper bound network
            net = conslearn.monotonic.buildPMNN(InputSizeSet.Input, 10);

            % Test that the network has the expected input layer
            testCase.verifyThat(net.Layers(1), IsOfClass(InputSizeSet.ExpectedLayer));
        end

        function verifyNetworkLengthIsCorrect(testCase, FullyConnectedLayerSizesSet)
            import matlab.unittest.constraints.HasLength

            % Build Lipschitz upper bound network
            numberOfChannels = 20;
            net = conslearn.monotonic.buildPMNN( ...
                numberOfChannels, ...
                FullyConnectedLayerSizesSet);

            % Test that the network has the correct number of layers
            % (1 input, N activation, N fullyconnect, 1 addition and 1
            % residual monotonic layer, where N is the length of the
            % numHiddenUnits input)
            N = length(FullyConnectedLayerSizesSet);
            testCase.verifyThat(net.Layers, HasLength(2*N + 3));
        end

        function verifyFullyConnectedLayersAreCorrect(testCase, FullyConnectedLayerSizesSet)
            import matlab.unittest.constraints.IsEqualTo

            % Build Lipschitz upper bound network
            numberOfChannels = 20;
            net = conslearn.monotonic.buildPMNN( ...
                numberOfChannels, ...
                FullyConnectedLayerSizesSet);

            % Get indices for hidden fully connected layers
            fcLayerIdx = iFindLayerIdxWithType(net, 'nnet.cnn.layer.FullyConnectedLayer');

            % Test that the hidden inputs are assigned correctly to the
            % fully connected layers
            testCase.verifyThat([net.Layers(fcLayerIdx).OutputSize], ...
                IsEqualTo(FullyConnectedLayerSizesSet));
        end

        function verifyActivationLayersAreCorrect(testCase, FullyConnectedLayerSizesSet, ActivationFunctionSet)
            import matlab.unittest.constraints.HasLength

            % Build Lipschitz upper bound network
            numberOfChannels = 20;
            net = conslearn.monotonic.buildPMNN( ...
                numberOfChannels, ...
                FullyConnectedLayerSizesSet, ...
                Activation=ActivationFunctionSet.Input);

            activationLayerIdx = iFindLayerIdxWithType(net, ActivationFunctionSet.ExpectedLayerClass);

            testCase.verifyThat(activationLayerIdx, HasLength(length(FullyConnectedLayerSizesSet)));
        end

        function verifyResidualMonotonicLayerIsCorrect(testCase, MonotonicChannelIdxSet, MonotonicTrendSet)
            import matlab.unittest.constraints.IsEmpty
            import matlab.unittest.constraints.IsEqualTo

            % Build Lipschitz upper bound network
            numberOfChannels = 20;
            numberOfHiddenUnits = [10, 50, 1000, 10];
            net = conslearn.monotonic.buildPMNN( ...
                numberOfChannels, ...
                numberOfHiddenUnits, ...
                MonotonicChannelIdx=MonotonicChannelIdxSet, ...
                MonotonicTrend=MonotonicTrendSet.Input);

            % Test that the neural network has a
            % conslearn.layer.ResidualMonotonicLayer
            residualMonotonicLayerIdx = iFindLayerIdxWithType(net, 'conslearn.layer.ResidualMonotonicLayer');
            testCase.verifyThat(residualMonotonicLayerIdx, ~IsEmpty)

            % Test that the number of monotonic channels in the residual
            % monotonic layer are equal to all the channels of the input layer.
            testCase.verifyThat(net.Layers(residualMonotonicLayerIdx).MonotonicChannels, ...
                IsEqualTo(MonotonicChannelIdxSet));

            % Test the residual monotonic layer has the correct lambda
            % according to the monotonic trend
            testCase.verifyThat(net.Layers(residualMonotonicLayerIdx).ResidualScaling, ...
                IsEqualTo(MonotonicTrendSet.ExpectedResidualScaling));
        end

        function verifyAdditionLayerIsCorrect(testCase, FullyConnectedLayerSizesSet)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.IsOfClass

            % Build Lipschitz upper bound network
            numberOfChannels = 20;
            net = conslearn.monotonic.buildPMNN( ...
                numberOfChannels, ...
                FullyConnectedLayerSizesSet);

            % Test that the neural network has an addition layer at the end
            additionLayerIdx = iFindLayerIdxWithType(net, 'nnet.cnn.layer.AdditionLayer');
            testCase.verifyThat(additionLayerIdx, IsEqualTo(length(net.Layers)));

            % Get the layers connected to the inputs of the addition layer.
            % This is done to test the connections to the addition layer
            % are correct.
            additionLayerInputNames = net.Layers(additionLayerIdx).InputNames;
            additionLayerName = net.Layers(additionLayerIdx).Name;
            networkConnections = net.Connections;
            [~, additionLayerConnectionsIdx] = ismember(strcat(additionLayerName, '/', additionLayerInputNames), networkConnections.Destination);
            additionLayerSourceNames = networkConnections.Source(additionLayerConnectionsIdx);
            additionLayerSourceIdx = iFindLayerIdxWithName(net, additionLayerSourceNames);
            additionLayerSourceLayers = net.Layers(additionLayerSourceIdx);

            % Test that these layers are one fully connected layer and one
            % conslearn.layer.ResidualMonotonicLayer
            testCase.verifyThat(additionLayerSourceLayers(1), ...
                IsOfClass(?nnet.cnn.layer.FullyConnectedLayer))
            testCase.verifyThat(additionLayerSourceLayers(2), ...
                IsOfClass(?conslearn.layer.ResidualMonotonicLayer))
        end

        function verifyNetworkHasACorrectLipschitzUpperBound(testCase, PNormSet, ResidualScalingSet)
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.RelativeTolerance
            import matlab.unittest.constraints.IsLessThan

            % Build Lipschitz upper bound network
            net = conslearn.monotonic.buildPMNN(10, [10, 50, 1000, 10], ...
                ResidualScaling=ResidualScalingSet, ...
                pNorm=PNormSet);

            % Calculate Lipschitz upper bound
            actualLipschitzUpperBound = lipschitzUpperBound(net, PNormSet);
            expectedLipschitzUpperBound = ResidualScalingSet;

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

function idx = iFindLayerIdxWithName(net, name)
idx = find(contains({net.Layers.Name}, name));
end
