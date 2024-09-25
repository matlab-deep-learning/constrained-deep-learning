function net = buildPICNN(inputSize, numHiddenUnits, options)
% BUILDPICNN    Construct a partially input convex neural network (PICNN).
%
%   NET = BUILDPICNN(INPUTSIZE, NUMHIDDENUNITS) creates an initialized
%   dlnetwork object, NET, with either a featureInputLayer or an
%   imageInputLayer, depending on whether INPUTSIZE is a scalar or a vector
%   with 3 elements. NUMHIDDENUNITS is a vector of integers that
%   corresponds to the number of activations in the fully connected layers
%   in the network.
%
%   NET = BUILDPICNN(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   BUILDPICNN name-value arguments:
%
%   'ConvexNonDecreasingActivation'   - Specify the convex,
%                                       non-decreasing activation functions. 
%                                       The options are 'softplus' or 'relu'. 
%                                       The default is 'softplus'.
%   'Activation'                      - Specify the unconstrained activation
%                                       function for evolution of the state-like 
%                                       variable in the network. The options
%                                       are 'tanh', 'relu' or 'fullsort'. 
%                                       The default is 'tanh'.
%   'ConvexChannelIdx'                - Specify the channel indices for the
%                                       inputs that carry convex dependency
%                                       with the output, specified as a
%                                       vector of positive integers. For
%                                       image inputs, the convex channel
%                                       indices correspond to the indices
%                                       in the flattened image input. The
%                                       default value is 1.
% 
% The construction of this network corresponds to Eq 3 in [1] with the
% exception that the application of the convex, non-decreasing activation
% function on the network output is not applied. This maintains convexity
% but permits positive and negative network outputs. Additionally, and in
% keeping with the notation used in the reference, in this implementation
% of the network construction, for the u_i 'state-like' evolution, the
% number of hidden units in the fully connected operations for 'state-like'
% evolution is taken to be the same as the number of hidden units in the
% fully connected operations for z_i 'output-like' evolution. This
% restriction can be relaxed.
% 
% [1] Amos, Brandon, et al. Input Convex Neural Networks. arXiv:1609.07152,
% arXiv, 14 June 2017. arXiv.org, https://doi.org/10.48550/arXiv.1609.07152.

%   Copyright 2024 The MathWorks, Inc.

arguments
    inputSize (1,:) {iValidateInputSize(inputSize)}
    numHiddenUnits (1,:)
    options.ConvexNonDecreasingActivation = 'softplus'
    options.Activation = 'tanh'
    options.ConvexChannelIdx = 1
end

% Get hyperparameters for network construction
depth = numel(numHiddenUnits);
convexChannels = options.ConvexChannelIdx;
nonConvexChannels = 1:prod(inputSize);
nonConvexChannels(options.ConvexChannelIdx) = [];
convexInputSize = numel(convexChannels);

% Prepare the two types of valid activation functions
switch options.ConvexNonDecreasingActivation
    case 'relu'
        pndFcn = @(k)reluLayer(Name="pnd_" + k);
    case 'softplus'
        pndFcn = @(k)softplusLayer(Name="pnd_" + k);
end
switch options.Activation
    case 'fullsort'
        ncaFcn = @(k)iFullSortLayer("nca_" + k);
    case 'relu'
        ncaFcn = @(k)reluLayer(Name="nca_" + k);
    case 'tanh'
        ncaFcn = @(k)tanhLayer(Name="nca_" + k);
end

%% Construct the correct input layers and add to layer graph
if isequal(numel(inputSize),1)
    inputLayer = [featureInputLayer(inputSize,Name='input',Normalization='none')];
elseif isequal(numel(inputSize),3)
    inputLayer = [imageInputLayer(inputSize,Name='image_input',Normalization='none')
        flattenLayer(Name='input')];
end
inputConvexFilterLayer = conslearn.layer.FilterInputLayer(convexChannels,'convexIn');
inputNonConvexFilterLayer = conslearn.layer.FilterInputLayer(nonConvexChannels,'nonconvexIn');
lgraph = layerGraph(inputLayer);
lgraph = addLayers(lgraph, inputConvexFilterLayer);
lgraph = addLayers(lgraph, inputNonConvexFilterLayer);
lgraph = connectLayers(lgraph,"input","convexIn");
lgraph = connectLayers(lgraph,"input","nonconvexIn");

%% Add the layers for z1 'output' evolution (no u 'state' evolution required for depth 1)
% fullyConnected
lgraph = addLayers(lgraph,fullyConnectedLayer(convexInputSize,Name="fc_yu_0"));
lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(1),Name="fc_y_0"));
lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(1),Name="fc_u_0"));
% hadamard product
lgraph = addLayers(lgraph,multiplicationLayer(2,Name="mult_y_u_0"));
% addition
lgraph = addLayers(lgraph,additionLayer(2,Name="add_y_u_0"));

%% Connect all layers for z 'output' evolution
% 'y' add branch
lgraph = connectLayers(lgraph,"nonconvexIn","fc_yu_0"); % W_0^(yu) x + b_0^(y)
lgraph = connectLayers(lgraph,"convexIn","mult_y_u_0/in1"); % y .* ...
lgraph = connectLayers(lgraph,"fc_yu_0","mult_y_u_0/in2"); % ... W_0^(yu) x + b_0^(y)
lgraph = connectLayers(lgraph,"mult_y_u_0","fc_y_0"); % W_0^(y) * ( y .* ( W_0^(yu) x + b_0^(y) ) )
% 'u' add branch
lgraph = connectLayers(lgraph,"nonconvexIn","fc_u_0"); % W_0^(u) x + b_0
% Connect add branch and pass through activation
lgraph = connectLayers(lgraph,"fc_y_0","add_y_u_0/in1"); % W_0^(y) * ( y .* ( W_0^(yu) x + b_0^(y) ) ) + ...
lgraph = connectLayers(lgraph,"fc_u_0","add_y_u_0/in2"); % ... W_0^(u) x + b_0

for ii = 2:depth
    %% Add the layers for u 'state' evolution
    % fullyConnected
    lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(ii),Name="fcU_" + (ii-2)));
    % non-convex activation
    lgraph = addLayers(lgraph,ncaFcn(ii-2));

    %% Add the layers for z 'output' evolution
    % convex non-decreasing activation
    lgraph = addLayers(lgraph,pndFcn(ii-2));
    % fullyConnected
    lgraph = addLayers(lgraph, fullyConnectedLayer(convexInputSize,Name="fc_yu_" + (ii-1)));
    lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(ii-1),Name="fc_zu_" + (ii-1))); % has to have same outputsize as z_i
    lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(ii),Name="fc_y_" + (ii-1)));
    lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(ii),Name="fc_u_" + (ii-1)));
    lgraph = addLayers(lgraph,fullyConnectedLayer(numHiddenUnits(ii),Name="fc_z_+_" + (ii-1)));
    % relu (positivity enforcing layer)
    lgraph = addLayers(lgraph,reluLayer(Name="+_" + (ii-1)));
    % hadamard product
    lgraph = addLayers(lgraph,multiplicationLayer(2,Name="mult_z_u_" + (ii-1)));
    lgraph = addLayers(lgraph,multiplicationLayer(2,Name="mult_y_u_" + (ii-1)));
    % addition
    lgraph = addLayers(lgraph,additionLayer(3,Name="add_z_y_u_" + (ii-1)));

    %% Connect all layers for u 'state' evolution
    if isequal(ii,2)
        % Special case for connecting the input layer for the first time
        lgraph = connectLayers(lgraph,"nonconvexIn","fcU_" + (ii-2)); % W_i^(tilde) u_i + b_i^(tilde)
    else
        lgraph = connectLayers(lgraph,"nca_" + (ii-3),"fcU_" + (ii-2)); % W_i^(tilde) u_i + b_i^(tilde)
    end
    lgraph = connectLayers(lgraph,"fcU_" + (ii-2),"nca_" + (ii-2)); % g_i^(tilde) ( W_i^(tilde) u_i + b_i^(tilde) )

    %% Connect all layers for z 'output' evolution
    % Connect a convex non-decreasing activation to the z(ii-1) output
    if isequal(ii,2)
        % g_0 ( W_0^(y) * ( y .* ( W_0^(yu) x + b_0^(y) ) ) + W_0^(u) x + b_0 )
        lgraph = connectLayers(lgraph,"add_y_u_" + (ii-2),"pnd_" + (ii-2));
    else
        % g_i ( W_i^(z) * ( z_i .* [W_i^(zu) u_i + b_i^(z)]_+ ) + W_i^(y) * ( y .* ( W_i^(yu) u_i + b_i^(y) ) ) + W_i^(u) u_i + b_i )
        lgraph = connectLayers(lgraph,"add_z_y_u_" + (ii-2) ,"pnd_" + (ii-2));
    end

    % 'z' add branch
    lgraph = connectLayers(lgraph,"nca_" + (ii-2),"fc_zu_" + (ii-1)); % W_i^(zu) u_i + b_i^(z)
    lgraph = connectLayers(lgraph,"fc_zu_" + (ii-1),"+_" + (ii-1)); % [W_i^(zu) u_i + b_i^(z)]_+
    lgraph = connectLayers(lgraph,"pnd_" + (ii-2),"mult_z_u_"  + (ii-1) +"/in1"); % z_i .* ...
    lgraph = connectLayers(lgraph,"+_" + (ii-1),"mult_z_u_"  + (ii-1) +"/in2"); % ... [W_i^(zu) u_i + b_i^(z)]_+
    lgraph = connectLayers(lgraph,"mult_z_u_" + (ii-1),"fc_z_+_" + (ii-1)); % W_i^(z) * ( z_i .* [W_i^(zu) u_i + b_i^(z)]_+ )
    % 'y' add branch
    lgraph = connectLayers(lgraph,"nca_" + (ii-2),"fc_yu_" + (ii-1)); % W_i^(yu) u_i + b_i^(y)
    lgraph = connectLayers(lgraph,"convexIn","mult_y_u_"  + (ii-1) +"/in1"); % y .* ...
    lgraph = connectLayers(lgraph,"fc_yu_" + (ii-1),"mult_y_u_"  + (ii-1) +"/in2"); % ... W_i^(yu) u_i + b_i^(y)
    lgraph = connectLayers(lgraph,"mult_y_u_" + (ii-1),"fc_y_" + (ii-1)); % W_i^(y) * ( y .* ( W_i^(yu) u_i + b_i^(y) ) )
    % 'u' add branch
    lgraph = connectLayers(lgraph,"nca_" + (ii-2),"fc_u_" + (ii-1)); % W_i^(u) u_i + b_i
    % Connect add branch and pass through activation
    lgraph = connectLayers(lgraph,"fc_z_+_" + (ii-1),"add_z_y_u_" + (ii-1) + "/in1"); % W_i^(z) * ( z_i .* [W_i^(zu) u_i + b_i^(z)]_+ ) + ...
    lgraph = connectLayers(lgraph,"fc_y_" + (ii-1),"add_z_y_u_" + (ii-1) + "/in2"); % ... W_i^(y) * ( y .* ( W_i^(yu) u_i + b_i^(y) ) ) + ...
    lgraph = connectLayers(lgraph,"fc_u_" + (ii-1),"add_z_y_u_" + (ii-1) + "/in3"); % ... W_i^(u) u_i + b_i
end

% Initialize dlnetwork
net = dlnetwork(lgraph);
net = conslearn.convex.makeNetworkConvex(net);
end

function layer = iFullSortLayer(k)
layer = conslearn.layer.FullSortLayer(k);
end

function iValidateInputSize(inputSize)
if prod(inputSize) <= 1
    error("For a partially-convex network, you must have at least 2 input channels.")
end
end