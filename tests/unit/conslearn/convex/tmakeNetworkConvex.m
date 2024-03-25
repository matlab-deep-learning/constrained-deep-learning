classdef tmakeNetworkConvex < matlab.unittest.TestCase
    
    % Copyright 2024 The MathWorks, Inc.

    methods (Test)
        function verifyLearnablesAreNonNegativeAfterMakingNetworkConvex(testCase)
            import matlab.unittest.constraints.*

            net = iCreateSimpleMLP;

            testCase.assumeThat(iNumberOfNegativeLearnables(net), IsGreaterThan(1));

            convexNet = conslearn.convex.makeNetworkConvex(net);

            testCase.verifyThat(iNumberOfNegativeLearnables(convexNet), IsEqualTo(0));
        end
    end
end

function net = iCreateSimpleMLP()
net = dlnetwork([ ...
    imageInputLayer([28 28]), ...
    fullyConnectedLayer(10, Name="fc_+_1"), ...
    reluLayer, ...
    fullyConnectedLayer(10, Name="fc_+_2"), ...
    reluLayer]);
end

function out = iNumberOfNegativeLearnables(net)
out = sum(cellfun(@(x) nnz(extractdata(x<0)),net.Learnables.Value), "all");
end
