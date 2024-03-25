function net = buildFMNN(inputSize, numHiddenUnits, options)
% BUILDFMNN    Construct a fully input monotonic neural network (FMNN).
%
%   NET = BUILDFMNN(INPUTSIZE, NUMHIDDENUNITS) creates an initialized
%   dlnetwork object, NET, with either a featureInputLayer or an
%   imageInputLayer, depending on whether INPUTSIZE is a scalar or a vector
%   with 3 elements. NUMHIDDENUNITS is a vector of integers that
%   corresponds to the number of activations in the fully connected layers
%   in the network.
%
%   NET = BUILDFMNN(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDFMNN name-value arguments:
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
% 
% The construction of this network follows the discussion in [1].
% 
% [1] Kitouni, Ouail, et al. Expressive Monotonic Neural Networks.
% arXiv:2307.07512, arXiv, 14 July 2023. arXiv.org,
% http://arxiv.org/abs/2307.07512.

%   Copyright 2024 The MathWorks, Inc.

arguments
    inputSize (1,:)
    numHiddenUnits (1,:)
    options.ResidualScaling = 1
    options.Activation = "fullsort"
    options.MonotonicTrend = "increasing"
    options.pNorm = Inf
end
% Monotonic in all channels
monotonicChannels = 1:inputSize(end);

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