classdef tmakeParametersLipschitz < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidInputs = iCreateTestParameter();
    end
    
    methods(Test)
        function verifyCorrectResultForSimpleCase(testCase, ValidInputs)
            out = conslearn.lipschitz.makeParametersLipschitz( ...
                ValidInputs.W, ValidInputs.lambda, ValidInputs.p);
            
            testCase.verifyEqual(out, ValidInputs.ExpectedOutput);
        end
    end
    
end

function param = iCreateTestParameter()
w = dlarray([1, 3, 5]);
param.p1 = struct( ...
    W = w, ...
    p = 1, ...
    lambda = 2, ...
    ExpectedOutput = w/2.5);
param.p2 = struct( ...
    W = w, ...
    p = 2, ...
    lambda = 2, ...
    ExpectedOutput = w/(norm(extractdata(w),2)/2));
param.pinf = struct( ...
    W = w, ...
    p = inf, ...
    lambda = 2, ...
    ExpectedOutput = w/4.5);
param.pnormDividedByLambdaSmallerThanOne = struct( ...
    W = w, ...
    p = 1, ...
    lambda = 10, ...
    ExpectedOutput = w);
end