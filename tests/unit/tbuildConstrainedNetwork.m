classdef tbuildConstrainedNetwork < matlab.unittest.TestCase

    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        ValidConstraintSet = iCreateValidConstraintSet;
        ValidInputSizeSet = iCreateValidInputSizeSet;
        ValidHiddenUnitsSet = iCreateValidHiddenUnitsSet;
        ValidPndActivationSet = iCreateValidPndActivationSet;
        ValidConvexChannelIdxSet = iCreateValidConvexChannelIdxSet;
        ValidPNormSet = iCreateValidPNormSet;
        ValidMonotonicChannelIdxSet = iCreateValidMonotonicChannelIdxSet;
        ValidMonotonicTrendSet = iCreateValidMonotonicTrendSet;
        ValidUpperBoundLipschitzConstantSet = iCreateValidUpperBoundLipschitzConstantSet;
        ValidResidualScalingSet = iCreateValidUpperBoundLipschitzConstantSet; % This can be considered the same as UpperBoundLipschitzConstantSet
        ValidActivationSet = iCreateValidActivationSet;

        InvalidConstraintSet = {"hello", 1};
        InvalidInputSizeSet = {[28 28 28 3], [28 3]};
        InvalidOptionalParamsFullyConvex = iCreateInvalidOptionalParamsFullyConvex;
        InvalidOptionalParamsPartiallyConvex = iCreateInvalidOptionalParamsPartiallyConvex;
        InvalidOptionalParamsFullyMonotonic = iCreateInvalidOptionalParamsFullyMonotonic;
        InvalidOptionalParamsPartiallyMonotonic = iCreateInvalidOptionalParamsPartiallyMonotonic;
        InvalidOptionalParamsLipschitzConstrained = iCreateInvalidOptionalParamsLipschitzConstrained;
    end

    methods (Test)
        % Tests to verify we can build networks with various combinations
        % of valid inputs parameters

        function canBuildNetworkWithDefaultOptionalInputs(testCase, ValidConstraintSet, ValidInputSizeSet, ValidHiddenUnitsSet)
            constraint = ValidConstraintSet;
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end

        function canBuildFullyConvexNetworkWithOptionalInputs(testCase, ValidInputSizeSet, ValidHiddenUnitsSet, ValidPndActivationSet)
            constraint = "fully-convex";
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;
            pndActivation = ValidPndActivationSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                "PositiveNonDecreasingActivation", pndActivation);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end

        function canBuildPartiallyConvexNetworkWithOptionalInputs(testCase, ValidInputSizeSet, ValidHiddenUnitsSet, ValidPndActivationSet, ValidConvexChannelIdxSet, ValidActivationSet)
            constraint = "partially-convex";
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;
            pndActivation = ValidPndActivationSet;
            activation = ValidActivationSet;
            convexChannelIdx = ValidConvexChannelIdxSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                "PositiveNonDecreasingActivation", pndActivation, ...
                "Activation", activation, ...
                "ConvexChannelIdx", convexChannelIdx);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end

        function canBuildFullyMonotonicNetworkWithOptionalInputs(testCase, ValidInputSizeSet, ValidHiddenUnitsSet, ValidMonotonicTrendSet, ValidResidualScalingSet, ValidActivationSet, ValidPNormSet)
            constraint = "fully-monotonic";
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;
            monotonicTrend = ValidMonotonicTrendSet;
            residualScaling = ValidResidualScalingSet;
            activation = ValidActivationSet;
            pnorm = ValidPNormSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                "MonotonicTrend", monotonicTrend, ...
                "ResidualScaling", residualScaling, ...
                "Activation", activation,...
                "pNorm", pnorm);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end

        function canBuildPartiallyMonotonicNetworkWithOptionalInputs(testCase, ValidInputSizeSet, ValidHiddenUnitsSet, ValidMonotonicTrendSet, ValidResidualScalingSet, ValidActivationSet, ValidPNormSet, ValidMonotonicChannelIdxSet)
            constraint = "partially-monotonic";
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;
            monotonicTrend = ValidMonotonicTrendSet;
            residualScaling = ValidResidualScalingSet;
            activation = ValidActivationSet;
            monotonicChannelIdx = ValidMonotonicChannelIdxSet;
            pnorm = ValidPNormSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                "MonotonicTrend", monotonicTrend, ...
                "MonotonicChannelIdx", monotonicChannelIdx, ...
                "ResidualScaling", residualScaling, ...
                "Activation", activation,...
                "pNorm", pnorm);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end

        function canBuildLipschitzConstrainedNetworkWithOptionalInputs(testCase, ValidInputSizeSet, ValidHiddenUnitsSet, ValidUpperBoundLipschitzConstantSet, ValidActivationSet, ValidPNormSet)
            constraint = "lipschitz";
            inputSize = ValidInputSizeSet;
            numHiddenUnits = ValidHiddenUnitsSet;
            pnorm = ValidPNormSet;
            upperBoundLipschitzConstant = ValidUpperBoundLipschitzConstantSet;
            activation = ValidActivationSet;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                "pNorm", pnorm, ...
                "UpperBoundLipschitzConstant", upperBoundLipschitzConstant, ...
                "Activation", activation);

            net = testCase.verifyWarningFree(fcn);

            testCase.verifyClass(net, "dlnetwork");
        end
    end

    methods(Test)
        % Negative tests to verify we error on non-valid inputs

        function errorsForInvalidConstraint(testCase, InvalidConstraintSet)
            import matlab.unittest.constraints.Throws

            constraint = InvalidConstraintSet;
            inputSize = 10;
            numHiddenUnits = [128 10];

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForInvalidInputSize(testCase, InvalidInputSizeSet)
            import matlab.unittest.constraints.Throws

            constraint = "fully-monotonic";
            inputSize = InvalidInputSizeSet;
            numHiddenUnits = [128 10];

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForFullyConvexAndInvalidNameValuePairs(testCase, InvalidOptionalParamsFullyConvex)
            import matlab.unittest.constraints.Throws

            constraint = "fully-convex";
            inputSize = 10;
            numHiddenUnits = [128 10];
            invalidName = InvalidOptionalParamsFullyConvex.Name;
            invalidValue = InvalidOptionalParamsFullyConvex.Value;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                invalidName, invalidValue);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForPartiallyConvexAndInvalidNameValuePairs(testCase, InvalidOptionalParamsPartiallyConvex)
            import matlab.unittest.constraints.Throws

            constraint = "partially-convex";
            inputSize = 10;
            numHiddenUnits = [128 10];
            invalidName = InvalidOptionalParamsPartiallyConvex.Name;
            invalidValue = InvalidOptionalParamsPartiallyConvex.Value;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                invalidName, invalidValue);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForFullyMonotonicAndInvalidNameValuePairs(testCase, InvalidOptionalParamsFullyMonotonic)
            import matlab.unittest.constraints.Throws

            constraint = "fully-monotonic";
            inputSize = 10;
            numHiddenUnits = [128 10];
            invalidName = InvalidOptionalParamsFullyMonotonic.Name;
            invalidValue = InvalidOptionalParamsFullyMonotonic.Value;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                invalidName, invalidValue);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForPartiallyMonotonicAndInvalidNameValuePairs(testCase, InvalidOptionalParamsPartiallyMonotonic)
            import matlab.unittest.constraints.Throws

            constraint = "fully-monotonic";
            inputSize = 10;
            numHiddenUnits = [128 10];
            invalidName = InvalidOptionalParamsPartiallyMonotonic.Name;
            invalidValue = InvalidOptionalParamsPartiallyMonotonic.Value;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                invalidName, invalidValue);

            testCase.verifyThat(fcn, Throws(?MException));
        end

        function errorsForLipschitzConstrainedAndInvalidNameValuePairs(testCase, InvalidOptionalParamsLipschitzConstrained)
            import matlab.unittest.constraints.Throws

            constraint = "lipschitz";
            inputSize = 10;
            numHiddenUnits = [128 10];
            invalidName = InvalidOptionalParamsLipschitzConstrained.Name;
            invalidValue = InvalidOptionalParamsLipschitzConstrained.Value;

            fcn = @() buildConstrainedNetwork(constraint, inputSize, numHiddenUnits, ...
                invalidName, invalidValue);

            testCase.verifyThat(fcn, Throws(?MException));
        end     
    end

end

function param = iCreateValidConstraintSet()
param.FullyConvex = "fully-convex";
param.PartiallyConvex = "partially-convex";
param.FullyMonotonic = "fully-monotonic";
param.PartiallyMonotonic = "partially-monotonic";
param.Lipschitz = "lipschitz";
end

function param = iCreateValidInputSizeSet()
param.FeatureInput = 10;
param.ImageInput = [28 28 3];
end

function param = iCreateValidHiddenUnitsSet()
param.TwoHiddenUnits = [128 10];
param.ThreeHiddenUnits = [256 128 10];
param.FourHiddenUnits = [512 256 128 10];
end

function param = iCreateValidPndActivationSet()
param.SoftPlus = "softplus";
param.ReLU = "relu";
end

function param = iCreateValidConvexChannelIdxSet()
param.ScalarOne = 1;
param.ScalarTen = 10;
param.VectorNonConsecutive = [1 2 10];
param.VectorConsecutive = 1:4;
end

function param = iCreateValidPNormSet()
param.One = 1;
param.Two = 2;
param.Inf = inf;
end

function param = iCreateValidMonotonicChannelIdxSet()
param.ScalarOne = 1;
param.ScalarTen = 10;
param.VectorNonConsecutive = [1 2 10];
param.VectorConsecutive = 1:4;
end

function param = iCreateValidMonotonicTrendSet()
param.Increasing = "increasing";
param.Decreasing = "decreasing";
end

function param = iCreateValidUpperBoundLipschitzConstantSet()
param.FloatingPoint = 1.5;
param.Integer = 4;
end

function param = iCreateValidActivationSet()
param.ReLU = "relu";
param.Tanh = "tanh";
param.FullSort = "fullsort";
end

function param = iCreateInvalidOptionalParamsFullyConvex()
param.Activation = struct( ...
    Name = "Activation", ...
    Value = "relu");

param.ConvexChannelIdx = struct( ...
    Name = "ConvexChannelIdx", ...
    Value = 1);

param.UpperBoundLipschitzConstant = struct( ...
    Name = "UpperBoundLipschitzConstant", ...
    Value = 1);

param.ResidualScaling = struct( ...
    Name = "ResidualScaling", ...
    Value = 1);

param.MonotonicTrend = struct( ...
    Name = "MonotonicTrend", ...
    Value = "decreasing");

param.MonotonicChannelIdx = struct( ...
    Name = "MonotonicChannelIdx", ...
    Value = 1);

param.pNorm = struct( ...
    Name = "pNorm", ...
    Value = 1);
end

function param = iCreateInvalidOptionalParamsPartiallyConvex()
param.ResidualScaling = struct( ...
    Name = "ResidualScaling", ...
    Value = 1);

param.UpperBoundLipschitzConstant = struct( ...
    Name = "UpperBoundLipschitzConstant", ...
    Value = 1);

param.MonotonicTrend = struct( ...
    Name = "MonotonicTrend", ...
    Value = "decreasing");

param.MonotonicChannelIdx = struct( ...
    Name = "MonotonicChannelIdx", ...
    Value = 1);

param.pNorm = struct( ...
    Name = "pNorm", ...
    Value = 1);
end

function param = iCreateInvalidOptionalParamsFullyMonotonic()
param.UpperBoundLipschitzConstant = struct( ...
    Name = "UpperBoundLipschitzConstant", ...
    Value = 1);

param.PositiveNonDecreasingActivation = struct( ...
    "Name", "PositiveNonDecreasingActivation", ...
    "Value", "relu");

param.ConvexChannelIdx = struct( ...
    Name = "ConvexChannelIdx", ...
    Value = 1);

param.MonotonicChannelIdx = struct( ...
    Name = "MonotonicChannelIdx", ...
    Value = 1);
end

function param = iCreateInvalidOptionalParamsPartiallyMonotonic()
param.UpperBoundLipschitzConstant = struct( ...
    Name = "UpperBoundLipschitzConstant", ...
    Value = 1);

param.PositiveNonDecreasingActivation = struct( ...
    "Name", "PositiveNonDecreasingActivation", ...
    "Value", "relu");

param.ConvexChannelIdx = struct( ...
    Name = "ConvexChannelIdx", ...
    Value = 1);
end

function param = iCreateInvalidOptionalParamsLipschitzConstrained()
param.ResidualScaling = struct( ...
    Name = "ResidualScaling", ...
    Value = 1);

param.PositiveNonDecreasingActivation = struct( ...
    "Name", "PositiveNonDecreasingActivation", ...
    "Value", "relu");

param.ConvexChannelIdx = struct( ...
    Name = "ConvexChannelIdx", ...
    Value = 1);

param.MonotonicChannelIdx = struct( ...
    Name = "MonotonicChannelIdx", ...
    Value = 1);

param.MonotonicTrend = struct( ...
    Name = "MonotonicTrend", ...
    Value = "decreasing");
end