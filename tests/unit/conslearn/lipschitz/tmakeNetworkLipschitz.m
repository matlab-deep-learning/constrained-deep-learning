classdef tmakeNetworkLipschitz < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    methods(Test)
        function verifyNetworkBecomesLipschitzConstrained(testCase)
            net = dlnetwork([ ...
                featureInputLayer(10), ...
                reluLayer, ...
                fullyConnectedLayer(10)]);
            pnorm = 1;
            lipschitzConstant = 2;

            % Get the learnables and lipschitz upper bound from before
            % making it Lipschitz constrained
            learnablesBefore = net.Learnables.Value;
            lipschitzUpperBoundBefore = lipschitzUpperBound(net, pnorm);

            % Make network Lipschitz constrained
            net = conslearn.lipschitz.makeNetworkLipschitz(net, pnorm, lipschitzConstant);

            % Get the learnables and lipschitz upper bound after making it
            % Lipschitz constrained
            learnablesAfter = net.Learnables.Value;
            lipschitzUpperBoundAfter = lipschitzUpperBound(net, pnorm);

            % Verify that the learnables before and after are not equal
            testCase.verifyNotEqual(learnablesAfter, learnablesBefore);

            % Verify that the lipschitz upper bound after conversion is
            % lower than before conversion
            testCase.verifyLessThan(lipschitzUpperBoundAfter, lipschitzUpperBoundBefore);
        end
    end
end