classdef tFullSortLayer < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.
    
    properties(TestParameter)
        ValidInputsAndOutputs = iCreateValidInputsAndOutputs;
    end

    methods (Test)
        function verifyConstructorSetsLayerPropertiesCorrectly(testCase)
            layer = conslearn.layer.FullSortLayer("sort");

            testCase.verifyEqual(layer.Name, 'sort');
        end

        function verifyPredictWorksAsExpected(testCase, ValidInputsAndOutputs)
            layer = conslearn.layer.FullSortLayer("sort");

            ActualOutput = layer.predict(ValidInputsAndOutputs.Input);

            testCase.verifyEqual(ActualOutput, ValidInputsAndOutputs.ExpectedOutput)
        end
    end

end

function param = iCreateValidInputsAndOutputs()
param.dlarrayOneObservationCB = struct( ...
    Input = dlarray([4, 3, 2, 1]', "CB"), ...
    ExpectedOutput = dlarray([1, 2, 3, 4]', "CB"));

param.dlarrayThreeObservationsCB = struct( ...
    Input = dlarray([ ...
    13, 12, 10, 9; ...
    19, 10, 3, 5; ...
    5, 4, 2, 10]', "CB"), ...
    ExpectedOutput = dlarray([ ...
    9, 10, 12, 13; ...
    3, 5, 10, 19; ...
    2, 4, 5, 10; ...
    ]', "CB"));
end
