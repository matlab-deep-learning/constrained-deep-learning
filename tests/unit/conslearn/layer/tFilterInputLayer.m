classdef tFilterInputLayer < matlab.unittest.TestCase
    
    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        PermittedChannelsSet = {1, 2:5, [2:3; 4:5]}
    end

    methods (Test)
        function verifyConstructorSetsLayerPropertiesCorrectly(testCase, PermittedChannelsSet)
            layer = conslearn.layer.FilterInputLayer(PermittedChannelsSet, "filter");

            testCase.verifyEqual(layer.PermittedChannels, PermittedChannelsSet);

            testCase.verifyEqual(layer.Name, 'filter');
        end

        function verifyPredictWorksAsExpected(testCase, PermittedChannelsSet)
            layer = conslearn.layer.FilterInputLayer(PermittedChannelsSet, "filter");

            X = magic(20);

            ExpectedOutput = X(PermittedChannelsSet, :);

            ActualOutput = layer.predict(X);

            testCase.verifyEqual(ExpectedOutput, ActualOutput)
        end
    end

end
