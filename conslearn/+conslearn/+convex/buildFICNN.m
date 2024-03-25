function net = buildFICNN(inputSize, numHiddenUnits, options)
% BUILDFICNN    Construct a fully input convex neural network (FICNN).
%
%   NET = BUILDFICNN(INPUTSIZE, NUMHIDDENUNITS) creates an initialized
%   dlnetwork object, NET, with either a featureInputLayer or an
%   imageInputLayer, depending on whether INPUTSIZE is a scalar or a vector
%   with 3 elements. NUMHIDDENUNITS is a vector of integers that
%   corresponds to the number of activations in the fully connected layers
%   in the network.
%
%   NET = BUILDFICNN(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDFICNN name-value arguments:
%
%   'PositiveNonDecreasingActivation' - Specify the positive, convex,
%                                       non-decreasing activation functions. 
%                                       The options are 'softplus' or 'relu'. 
%                                       The default is 'softplus'.
%
% The construction of this network corresponds to Eq 2 in [1] with the
% exception that the application of the positive, non-decreasing activation
% function on the network output is not applied. This maintains convexity
% but permits positive and negative network outputs.
% 
% [1] Amos, Brandon, et al. Input Convex Neural Networks. arXiv:1609.07152,
% arXiv, 14 June 2017. arXiv.org, https://doi.org/10.48550/arXiv.1609.07152.

%   Copyright 2024 The MathWorks, Inc.

arguments
    inputSize (1,:)
    numHiddenUnits (1,:)
    options.PositiveNonDecreasingActivation = 'softplus'
end

% Construct the correct input layer
if isequal(numel(inputSize),1)
    tempLayers = [featureInputLayer(inputSize,Name='input',Normalization='none')];
elseif isequal(numel(inputSize),3)
    tempLayers = [imageInputLayer(inputSize,Name='image_input',Normalization='none')
        flattenLayer(Name='input')];
end

% Loop over construction of hidden units
switch options.PositiveNonDecreasingActivation
    case 'relu'
        pndFcn = @(k)reluLayer(Name="pnd_" + k);
    case 'softplus'
        pndFcn = @(k)softplusLayer(Name="pnd_" + k);
end

depth = numel(numHiddenUnits);
% Construct the 'core' network
tempLayers = [tempLayers
    fullyConnectedLayer(numHiddenUnits(1),Name="fc_z_1")
    ];
for ii = 2:depth
    tempLayers = [tempLayers
        pndFcn(ii-1)
        fullyConnectedLayer(numHiddenUnits(ii),Name="fc_z_+_" + ii)
        additionLayer(2,Name="add_" + ii)
        ];
end

% Create layer graph
lgraph = layerGraph(tempLayers);

% Add a cascading residual connection
for ii = 2:depth
    tempLayers = fullyConnectedLayer(numHiddenUnits(ii),Name="fc_y_+_" + ii);
    lgraph = addLayers(lgraph,tempLayers);
    lgraph = connectLayers(lgraph,"input","fc_y_+_" + ii);
    lgraph = connectLayers(lgraph,"fc_y_+_" + ii,"add_" + ii + "/in2");
end

% Initialize dlnetwork
net = dlnetwork(lgraph);
net = conslearn.convex.makeNetworkConvex(net);
end