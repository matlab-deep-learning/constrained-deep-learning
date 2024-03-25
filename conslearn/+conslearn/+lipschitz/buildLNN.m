function net = buildLNN(inputSize,numHiddenUnits,options)
% BUILDLNN    Construct a neural network with a computable Lipschitz upper
% bound (LNN).
%
%   NET = BUILDLNN(INPUTSIZE, NUMHIDDENUNITS) creates an initialized
%   dlnetwork object, NET, with either a featureInputLayer or an
%   imageInputLayer, depending on whether INPUTSIZE is a scalar or a vector
%   with 3 elements. NUMHIDDENUNITS is a vector of integers that
%   corresponds to the number of activations in the fully connected layers
%   in the network.
%
%   NET = BUILDLNN(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDLNN name-value arguments:
%
%   'Activation'                      - Specify the activation
%                                       function in the network. The
%                                       options are 'tanh', 'relu' or
%                                       'fullsort'. The default is
%                                       'fullsort'.
%   'UpperBoundLipschitzConstant'     - Specify the upper bound on the
%                                       Lipschitz constant for the network,
%                                       as a positive real number. The
%                                       default value is 1.
%   'pNorm'                           - Specify the p-norm to measure
%                                       distance with respect to the
%                                       Lipschitz continuity definition.
%                                       The default value is 1.
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
    options.Activation = 'fullsort'
    options.UpperBoundLipschitzConstant = 1
    options.pNorm = 1
end
% Construct the correct input layer
if isequal(numel(inputSize),1)
    tempLayers = [featureInputLayer(inputSize,Name='input',Normalization='none')];
elseif isequal(numel(inputSize),3)
    tempLayers = [imageInputLayer(inputSize,Name='image_input',Normalization='none')
        flattenLayer(Name='input')];
end

% Specified upper bound on Lipschitz constant and p-norm
ubLipschitzConstant = options.UpperBoundLipschitzConstant;
pNorm = options.pNorm;

% Loop over construction of hidden units
switch options.Activation
    case 'fullsort'
        gnpFcn = @(k)iFullSortLayer("fullsort_" + k);
    case 'relu'
        gnpFcn = @(k)reluLayer(Name="relu_" + k);
    case 'tanh'
        gnpFcn = @(k)tanhLayer(Name="tanh" + k);
end

% Construct the network
tempLayers = [tempLayers
    gnpFcn(1)
    fullyConnectedLayer(numHiddenUnits(1),Name="fc_" + 1,WeightsInitializer="orthogonal")
    ];
for ii = 2:numel(numHiddenUnits)
    tempLayers = [tempLayers
        gnpFcn(ii)
        fullyConnectedLayer(numHiddenUnits(ii),Name="fc_" + ii,WeightsInitializer="orthogonal")
        ];
end

% Initialize dlnetwork
net = dlnetwork(tempLayers);
net = conslearn.lipschitz.makeNetworkLipschitz(net,pNorm,ubLipschitzConstant);
end

function layer = iFullSortLayer(k)
layer = conslearn.layer.FullSortLayer(k);
end
