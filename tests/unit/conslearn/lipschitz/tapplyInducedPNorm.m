classdef tapplyInducedPNorm < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidInputs = iCreateTestParameter();
    end
    
    methods(Test)
        function verifyCorrectResult(testCase, ValidInputs)
            out = conslearn.lipschitz.applyInducedPNorm( ...
                ValidInputs.W, ValidInputs.p);
            
            testCase.verifyEqual(extractdata(out), ValidInputs.ExpectedOutput);
        end
    end
    
end

function param = iCreateTestParameter()
w = dlarray([1, 3, 5]);
param.p1 = struct( ...
    W = w, ...
    p = 1, ...
    ExpectedOutput = 5);
param.p2 = struct( ...
    W = w, ...
    p = 2, ...
    ExpectedOutput = norm(extractdata(w),2));
param.pinf = struct( ...
    W = w, ...
    p = inf, ...
    ExpectedOutput = 9);
end