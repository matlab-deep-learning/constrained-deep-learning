classdef tmakeNetworkMonotonic < matlab.unittest.TestCase
    % Copyright 2024 The MathWorks, Inc.

    properties(TestParameter)
        PNormSet = {1, 2, Inf}
        ResidualScalingSet = {0.1, 1, 10}
    end

    methods(Test)
        function verifyNetworkBecomesMonotonic(testCase,PNormSet,ResidualScalingSet)
            lnet = dlnetwork([ ...
                featureInputLayer(10), ...
                reluLayer, ...
                fullyConnectedLayer(10)]);

            % Add ResidualLayer
            lgraph = layerGraph(lnet);
            residualScaling = ResidualScalingSet;
            tempLayers = conslearn.layer.ResidualMonotonicLayer(1:10, residualScaling);
            lgraph = addLayers(lgraph,tempLayers);

            % Add AdditionLayer
            tempLayers = additionLayer(2);
            lgraph = addLayers(lgraph,tempLayers);

            % Connect layers
            lgraph = connectLayers(lgraph,"input","res_mono");
            lgraph = connectLayers(lgraph,"fc" ,"addition/in1");
            lgraph = connectLayers(lgraph,"res_mono","addition/in2");

            % Initialize dlnetwork
            net = dlnetwork(lgraph);

            % Make network monotonic
            net = conslearn.monotonic.makeNetworkMonotonic(net,PNormSet);

            % Verify that the lipschitz upper bound after conversion is
            % lower than the ResidualScaling which guarantees monotonicity
            lipschitzUpperBoundAfter = lipschitzUpperBound(net, PNormSet);

            % Apply an admissibility constant, epsilon, to test for
            % monotonicity up to precision errors
            epsilon = 1e-5;
            testCase.verifyLessThan(lipschitzUpperBoundAfter, residualScaling + epsilon);
        end
    end
end