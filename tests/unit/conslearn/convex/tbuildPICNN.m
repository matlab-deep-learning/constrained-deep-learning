classdef tbuildPICNN < matlab.unittest.TestCase
    %   Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        FullyConnectedLayerSizesSet = struct( ...
            TwoFCLayersSameSize = [10, 10], ...
            FourFCLayersSameSize = [10, 10, 10, 10], ...
            TwoFCLayersDifferentSize = [10, 100], ...
            FourFCLayersDifferentSize = [10, 50, 1000, 10])

        PndActivationFunctionSet = struct( ...
            ReLU = struct( ...
            Input = 'relu', ...
            ExpectedLayer = ?nnet.cnn.layer.ReLULayer), ...
            Softplus = struct( ...
            Input = 'softplus', ...
            ExpectedLayer = ?rl.layer.SoftplusLayer));

        ConvexChannelIdxSet = {1, 2:4, [3:8 10]};

        UnconstrainedActivationFunctionSet = struct( ...
            Fullsort = struct( ...
            Input = 'fullsort', ...
            ExpectedLayer = ?conslearn.layer.FullSortLayer), ...
            ReLU = struct( ...
            Input = 'relu', ...
            ExpectedLayer = ?nnet.cnn.layer.ReLULayer), ...
            Tanh = struct( ...
            Input = 'tanh', ...
            ExpectedLayer = ?nnet.cnn.layer.TanhLayer));
    end

    methods(Test)
        function verifyNetworkArchitectureWithFeatureInput(testCase, FullyConnectedLayerSizesSet)
            % Specify network's input size and hidden units
            inputSize = 10;
            numHiddenUnits = FullyConnectedLayerSizesSet;

            % Build network
            net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits);

            testCase.verifyClass(net, 'dlnetwork');

            testCase.verifyClass(net.Layers(1), ?nnet.cnn.layer.FeatureInputLayer);

            testCase.verifyEqual(numel(net.Layers), 1 + 7 + 12 * (numel(numHiddenUnits)-1));
        end

        function verifyNetworkArchitectureWithImageInput(testCase, FullyConnectedLayerSizesSet)
            % Specify network's input size and hidden units
            inputSize = [28, 28, 1];
            numHiddenUnits = FullyConnectedLayerSizesSet;

            % Build network
            net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits);

            testCase.verifyClass(net, 'dlnetwork');

            testCase.verifyClass(net.Layers(1), ?nnet.cnn.layer.ImageInputLayer);

            testCase.verifyEqual(numel(net.Layers), 2 + 7 + 12 * (numel(numHiddenUnits)-1));
        end

        function verifyPndActivationLayersAreCorrect(testCase, FullyConnectedLayerSizesSet, PndActivationFunctionSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass

            % Specify network's input size and hidden units
            inputSize = 10;
            numHiddenUnits = FullyConnectedLayerSizesSet;

            % Build network
            net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits, ...
                PositiveNonDecreasingActivation=PndActivationFunctionSet.Input);

            % Get indices for activation layers
            pndLayerIdx = iFindLayerIdxWithName(net, "pnd");

            % Test that there are one less activation layers to fully
            % connected layers
            testCase.verifyThat(pndLayerIdx, ...
                HasLength(length(numHiddenUnits)-1));

            % Test that the activation layer is of the expected class
            testCase.verifyThat(net.Layers(pndLayerIdx), ...
                IsOfClass(PndActivationFunctionSet.ExpectedLayer));
        end

        function verifyFilterInputLayersAreCorrect(testCase, FullyConnectedLayerSizesSet, ConvexChannelIdxSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass
            import matlab.unittest.constraints.IsEqualTo

            % Specify network's input size and hidden units
            inputSize = 10;
            numHiddenUnits = FullyConnectedLayerSizesSet;

            % Build network
            net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits, ...
                ConvexChannelIdx=ConvexChannelIdxSet);

            % Test the non-convex input filter layer
            nonconvexLayerIdx = iFindLayerIdxWithName(net, "nonconvex");

            testCase.verifyThat(net.Layers(nonconvexLayerIdx), ...
                IsOfClass(?conslearn.layer.FilterInputLayer));

            testCase.verifyThat(net.Layers(nonconvexLayerIdx).PermittedChannels, ...
                IsEqualTo(setdiff(1:inputSize, ConvexChannelIdxSet)));

            % Test the convex input filter layer
            convexLayerIdx = iFindLayerIdxWithName(net, "convex");

            testCase.verifyThat(net.Layers(convexLayerIdx(1)), ...
                IsOfClass(?conslearn.layer.FilterInputLayer));

            testCase.verifyThat(net.Layers(convexLayerIdx(1)).PermittedChannels, ...
                IsEqualTo(ConvexChannelIdxSet));

        end

        function verifyUnconstrainedActivationLayerIsCorrect(testCase, FullyConnectedLayerSizesSet, UnconstrainedActivationFunctionSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass

            % Specify network's input size and hidden units
            inputSize = 10;
            numHiddenUnits = FullyConnectedLayerSizesSet;

            % Build network
            net = conslearn.convex.buildPICNN(inputSize, numHiddenUnits, ...
                Activation=UnconstrainedActivationFunctionSet.Input);

            % Get indices for activation layers
            pndLayerIdx = iFindLayerIdxWithName(net, "nca");

            % Test that there are one less activation layers to fully
            % connected layers
            testCase.verifyThat(pndLayerIdx, ...
                HasLength(length(numHiddenUnits)-1));

            % Test that the activation layer is of the expected class
            testCase.verifyThat(net.Layers(pndLayerIdx), ...
                IsOfClass(UnconstrainedActivationFunctionSet.ExpectedLayer));
        end
    end

end

function idx = iFindLayerIdxWithName(net, name)
idx = find(contains({net.Layers.Name}, name));
end