classdef tbuildFICNN < matlab.unittest.TestCase
    
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
            TwoFCLayersSameSize = [10, 10], ...
            FourFCLayersSameSize = [10, 10, 10, 10], ...
            TwoFCLayersDifferentSize = [10, 100], ...
            FourFCLayersDifferentSize = [10, 50, 1000, 10])

        ActivationFunctionSet = struct( ...
            ReLU = struct( ...
            Input = 'relu', ...
            ExpectedLayer = ?nnet.cnn.layer.ReLULayer), ...
            Softplus = struct( ...
            Input = 'softplus', ...
            ExpectedLayer = ?rl.layer.SoftplusLayer));
    end

    methods(Test)
        function verifyInputLayerIsCorrect(testCase, InputSizeSet)
            import matlab.unittest.constraints.IsOfClass

            % Build convex neural network
            net = conslearn.convex.buildFICNN(InputSizeSet.Input, 10);

            % Test that the network has the expected input layer
            testCase.verifyThat(net.Layers(1), IsOfClass(InputSizeSet.ExpectedLayer))
        end

        function verifyNetworkWithOneFullyConnectedLayer(testCase)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass
            import matlab.unittest.constraints.IsEqualTo

            % Build convex neural network
            net = conslearn.convex.buildFICNN([28, 28, 1], 10);
            
            % Test that the network has just three layers (input, flatten
            % and fullyconnect)
            testCase.verifyThat(net.Layers, HasLength(3));

            % Test that the final layer is a fully connected layer
            testCase.verifyThat(net.Layers(end), ...
                IsOfClass(?nnet.cnn.layer.FullyConnectedLayer));

            % Test that the final layer has the correct number of hidden
            % inputs
            testCase.verifyThat(net.Layers(end).OutputSize, ...
                IsEqualTo(10));
        end

        function verifyFullyConnectedLayersAreCorrect(testCase, FullyConnectedLayerSizesSet)
            import matlab.unittest.constraints.IsEqualTo

            % Build convex neural network
            net = conslearn.convex.buildFICNN([28, 28, 1], FullyConnectedLayerSizesSet);

            % Get indices for hidden fully connected layers
            fcLayerIdx = find(contains({net.Layers.Name}, "fc_z"));

            % Test that the hidden inputs are assigned correctly to the
            % fully connected layers
            testCase.verifyThat([net.Layers(fcLayerIdx).OutputSize], ...
                IsEqualTo(FullyConnectedLayerSizesSet));
        end

        function verifyNumberOfAdditionLayersAreCorrect(testCase, FullyConnectedLayerSizesSet)
            import matlab.unittest.constraints.HasLength
            import matlab.unittest.constraints.IsOfClass

            % Build convex neural network
            net = conslearn.convex.buildFICNN([28, 28, 1], FullyConnectedLayerSizesSet);

            % Get indices for addition layers
            addLayerIdx = find(contains({net.Layers.Name}, "add"));

            % Test that there are one less addition layers to fully
            % connected layers
            testCase.verifyThat(addLayerIdx, ...
                HasLength(length(FullyConnectedLayerSizesSet)-1));

            % Test that the addition layers have the correct class
            testCase.verifyThat(net.Layers(addLayerIdx), ...
                IsOfClass(?nnet.cnn.layer.AdditionLayer));
        end

        function verifyActivationLayersAreCorrect(testCase, FullyConnectedLayerSizesSet, ActivationFunctionSet)
            import matlab.unittest.constraints.HasLength 
            import matlab.unittest.constraints.IsOfClass

            % Build convex neural network
            net = conslearn.convex.buildFICNN([28, 28, 1], FullyConnectedLayerSizesSet, ...
                PositiveNonDecreasingActivation = ActivationFunctionSet.Input);

            % Get indices for activation layers
            pndLayerIdx = find(contains({net.Layers.Name}, "pnd"));

            % Test that there are one less activation layers to fully
            % connected layers
            testCase.verifyThat(pndLayerIdx, ...
                HasLength(length(FullyConnectedLayerSizesSet)-1));

            % Test that the activation layer is of the expected class
            testCase.verifyThat(net.Layers(pndLayerIdx), ...
                IsOfClass(ActivationFunctionSet.ExpectedLayer));
        end
    end
end
