classdef tlipschitzUpperBound < matlab.unittest.TestCase
    %   Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidPNormSet = {1, 2, inf}
    end
    
    methods(Test)       
        function verifyOutputIsANumericScalar(testCase, ValidPNormSet)
            inputSize = 10;
            constraint = "lipschitz";
            numHiddenUnits = [512 256 128 10];

            net = buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            pLipschitzConstant = lipschitzUpperBound(net, ValidPNormSet);

            testCase.verifySize(pLipschitzConstant, [1 1]);

            testCase.verifyClass(extractdata(pLipschitzConstant), ?single);
        end
    end
    
end