classdef tResidualMonotonicLayer < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        PermittedChannelsSet = {1, 2:5, [2:3; 4:5]}
    end

    methods (Test)
        function verifyConstructorSetsLayerPropertiesCorrectly(testCase)
            monotonicChannels = 1:3;
            lipschitzConstant = 2;

            layer = conslearn.layer.ResidualMonotonicLayer(monotonicChannels, lipschitzConstant);

            testCase.verifyEqual(layer.MonotonicChannels, monotonicChannels);

            testCase.verifyEqual(layer.ResidualScaling, lipschitzConstant);

            testCase.verifyEqual(layer.Name, 'res_mono');
        end

        function verifyPredictWorksAsExpected(testCase)
            monotonicChannels = 1:3;
            lipschitzConstant = 2;

            layer = conslearn.layer.ResidualMonotonicLayer(monotonicChannels, lipschitzConstant);

            X = magic(20);

            ExpectedOutput = lipschitzConstant*sum(X(monotonicChannels, :), 1);

            ActualOutput = layer.predict(X);

            testCase.verifyEqual(ExpectedOutput, ActualOutput)
        end
    end

end
