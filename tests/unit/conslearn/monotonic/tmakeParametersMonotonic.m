classdef tmakeParametersMonotonic < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidInputs = iCreateTestParameter();
    end

    methods(Test)
        function verifyCorrectResultForSimpleCase(testCase, ValidInputs)
            out = conslearn.monotonic.makeParametersMonotonic( ...
                ValidInputs.W, ValidInputs.lambda, ValidInputs.pNorm);

            testCase.verifyEqual(extractdata(out), extractdata(ValidInputs.ExpectedOutput), AbsTol=1e-12);
        end
    end

end

function param = iCreateTestParameter()
w = dlarray([1, 3, 5]);
param.pnormDividedByLambdaLargerThanOnePNormOne = struct( ...
    W = w, ...
    lambda = 2, ...
    pNorm = 1, ...
    ExpectedOutput = dlarray([1 2 2]));

param.pnormDividedByLambdaSmallerThanOnePNormOne = struct( ...
    W = w, ...
    lambda = 10, ...
    pNorm = 1, ...
    ExpectedOutput = w);

param.pnormDividedByLambdaLargerThanOnePNormTwo = struct( ...
    W = w, ...
    lambda = 2, ...
    pNorm = 2, ...
    ExpectedOutput = dlarray([0.338061701891407 1.014185105674220 1.690308509457033]));

param.pnormDividedByLambdaSmallerThanOnePNormTwo = struct( ...
    W = w, ...
    lambda = 10, ...
    pNorm = 2, ...
    ExpectedOutput = w);

param.pnormDividedByLambdaLargerThanOnePNormInf = struct( ...
    W = w, ...
    lambda = 2, ...
    pNorm = Inf, ...
    ExpectedOutput = dlarray([2/9 2/3 10/9]));

param.pnormDividedByLambdaSmallerThanOnePNormInf = struct( ...
    W = w, ...
    lambda = 10, ...
    pNorm = Inf, ...
    ExpectedOutput = w);
end