function net = buildPMNN(inputSize, numHiddenUnits, options)
% BUILDPMNN    Construct a partially input monotonic neural network (PMNN).
%
%   NET = BUILDPMNN(INPUTSIZE, NUMHIDDENUNITS) creates an initialized
%   dlnetwork object, NET, with either a featureInputLayer or an
%   imageInputLayer, depending on whether INPUTSIZE is a scalar or a vector
%   with 3 elements. NUMHIDDENUNITS is a vector of integers that
%   corresponds to the number of activations in the fully connected layers
%   in the network.
%
%   NET = BUILDPMNN(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDPMNN name-value arguments:
%
%   'Activation'                      - Specify the activation
%                                       function in the network. The
%                                       options are 'tanh', 'relu' or
%                                       'fullsort'. The default is
%                                       'fullsort'.
%   'ResidualScaling'                 - The scale factor applied to the sum
%                                       of the inputs that carry monotonic
%                                       dependency with the output.
%                                       The default value is 1.
%   'MonotonicTrend'                  - Specify the monotonic trend of the
%                                       output with respect to increasing
%                                       inputs, as either "increasing" or
%                                       "decreasing". The default is
%                                       "increasing".
%   'MonotonicChannelIdx'             - Specify the channel indices for the
%                                       inputs that carry monotonic dependency
%                                       with the output, specified as a
%                                       vector of positive integers. For
%                                       image inputs, the monotonic channel
%                                       indices correspond to the indices
%                                       in the flattened image input. The
%                                       default value is 1.
% 
% The construction of this network follows the discussion in [1].
% 
% [1] Kitouni, Ouail, et al. Expressive Monotonic Neural Networks.
% arXiv:2307.07512, arXiv, 14 July 2023. arXiv.org,
% http://arxiv.org/abs/2307.07512.

%   Copyright 2024 The MathWorks, Inc.

arguments
    inputSize (1,:) {iValidateInputSize(inputSize)}
    numHiddenUnits (1,:)
    options.ResidualScaling = 1
    options.Activation = 'fullsort'
    options.MonotonicTrend = "increasing"
    options.MonotonicChannelIdx = 1
    options.pNorm = Inf
end
% Monotonic in selected channels
monotonicChannels = options.MonotonicChannelIdx;

% Lipschitz signature
lipSignature = 1;
if isequal(options.MonotonicTrend,"decreasing")
    lipSignature = -1;
end

% Construct base Lipschitz network
lnet = conslearn.lipschitz.buildLNN(inputSize,numHiddenUnits,...
    Activation = options.Activation,...
    pNorm = options.pNorm);

% Specified upper bound on Lipschitz constant
ubLipschitzConstant = options.ResidualScaling;
ubLipschitzConstant = lipSignature*ubLipschitzConstant;

% Add ResidualLayer
lgraph = layerGraph(lnet);
tempLayers = iMonotonicLayer(monotonicChannels,ubLipschitzConstant);
lgraph = addLayers(lgraph,tempLayers);

% Add AdditionLayer
tempLayers = additionLayer(2,"Name","addition");
lgraph = addLayers(lgraph,tempLayers);

% Connect layers
lgraph = connectLayers(lgraph,"input","res_mono");
depth = numel(numHiddenUnits);
lgraph = connectLayers(lgraph,"fc_" + depth,"addition/in1");
lgraph = connectLayers(lgraph,"res_mono","addition/in2");

% Initialize dlnetwork
net = dlnetwork(lgraph);
net = conslearn.monotonic.makeNetworkMonotonic(net,options.pNorm);
end

function layer = iMonotonicLayer(monotonicChannels,lipschitzConstant)
layer = conslearn.layer.ResidualMonotonicLayer(monotonicChannels,lipschitzConstant);
end

function iValidateInputSize(inputSize)
if prod(inputSize) <= 1
    error("For a partially-monotonic network, you must have at least 2 input channels.")
end
end