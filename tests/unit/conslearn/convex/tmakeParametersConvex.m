classdef tmakeParametersConvex < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidInputsAndOutputs = iCreateValidInputsAndOutputs;
    end
    
    methods (Test)
        function verifyCorrectOutput(testCase, ValidInputsAndOutputs)
            input = ValidInputsAndOutputs.Input;
            expectedOutput = ValidInputsAndOutputs.ExpectedOutput;

            actualOutput = conslearn.convex.makeParametersConvex(input);

            testCase.verifyEqual(actualOutput, expectedOutput);
        end
    end

end

function param = iCreateValidInputsAndOutputs()
param.NegativeInput = struct( ...
    Input = dlarray(-10), ...
    ExpectedOutput = dlarray(0));
param.ZeroInput = struct( ...
    Input = dlarray(0), ...
    ExpectedOutput = dlarray(0));
param.PositiveInput = struct( ...
    Input = dlarray(2), ...
    ExpectedOutput = dlarray(2));
end