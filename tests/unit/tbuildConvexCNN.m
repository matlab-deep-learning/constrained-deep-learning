classdef tbuildConvexCNN < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.
    
    properties (TestParameter)
        SequenceInputSize = {1, 10, 25, 100};
        ImageInputSize = {[10 10], [10 25], [10 10 1], [10 25 3]};
        BadInputSize = struct( ...
            "Empty", struct("Value", [], "ErrID", "MATLAB:validators:mustBeNonempty"), ...
            "NonReal", struct("Value", 1i, "ErrID", "MATLAB:validators:mustBeReal"), ...
            "NonInteger", struct("Value", 2.5, "ErrID", "MATLAB:validators:mustBeInteger"), ...
            "NonPositive", struct("Value", -10, "ErrID", "MATLAB:validators:mustBePositive"), ...
            "TooLong", struct("Value", [5 5 3 1], "ErrID", ?MException));
    end

    methods (TestClassSetup)
        function addPathFixture(testCase)
            % Patch the internal dependencies
            import matlab.unittest.fixtures.PathFixture
            testCase.applyFixture(PathFixture("patches/", IncludeSubfolders=true));
        end
    end

    methods (Test)
        function verifySequenceInputSizeSwitch(testCase, SequenceInputSize)
            % Verify that a 1D input size calls into buildSequenceFICCNN
            fcnCalled = buildConvexCNN(SequenceInputSize);
            testCase.verifyEqual(fcnCalled, 'buildSequenceFICCNN');
        end

        function verifyImageInputSizeSwitch(testCase, ImageInputSize)
            % Verify that a 2D/3D input size calls into buildImageFICCNN
            fcnCalled = buildConvexCNN(ImageInputSize);
            testCase.verifyEqual(fcnCalled, 'buildImageFICCNN');
        end

        function verifyBadInputSizeError(testCase, BadInputSize)
            % Verify that bad input sizes error
            badInputCall = @() buildConvexCNN(BadInputSize.Value);
            testCase.verifyError(badInputCall, BadInputSize.ErrID);
        end
    end

end