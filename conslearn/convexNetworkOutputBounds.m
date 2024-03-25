function [netMin,netMax,netPred] = convexNetworkOutputBounds(net,Vn,options)
% CONVEXNETWORKOUTPUTBOUNDS    Compute the upper and lower bounds of a
% convex neural network over a hypercubic grid.
%
%   NETPRED = CONVEXNETWORKOUTPUTBOUNDS(NET, V) compute the predictions of
%   the convex network, NET, where NET is a dlnetwork object, on the
%   vertices of the hypercubic grid, V, where V is a cell array containing
%   the outputs of the NDGRID function for 1, 2 or 3 dimensions. NETPRED is
%   the same shape as an element of V, i.e., an NDGRID output shape. NET
%   must be a convex network, constructed using the
%   BUILDCONSTRAINEDNETWORK function with convex constraint, where the input
%   layer is a FeatureInputLayer with 3 or fewer input channels.
%
%   [NETMIN,NETMAX,NETPRED] = CONVEXNETWORKOUTPUTBOUNDS(NET, V) computes the
%   maximum and minimum of NET on the hypercubic regions specified by V.
%   NETMAX and NETMIN are the same shape as NETPRED but with 1 fewer
%   element in each dimension.
%
%   __ = CONVEXNETWORKOUTPUTBOUNDS(__,NAME=VALUE) specifies additional
%   options using one or more name-value arguments.
%
%   RefineLowerBounds   - Flag specifying to refine the network minimum
%                         bounds using Optimization Toolbox.
%                         When the minimum is not possible to obtain,
%                         NETMIN will contain NaN value.
%                         Setting RefineLowerBounds to true uses
%                         Optimization Toolbox to replace NaN
%                         values with correct lower bounds. 
%                         The default value is true.
%                         
%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork {iValidateNetwork(net)}
    Vn (:,1) cell {iValidateHypercubicVertices(Vn)}
    options.RefineLowerBounds (1,1) logical = true;
end

% Dimension of input domain
numInputs = net.Layers(1).InputSize;

% Preprocess vertices
Vn = iPreprocessVertices(Vn);

% Get number of network output channels
numOutputs = iComputeNumberChannelOutputs(net,Vn,numInputs);

% Compute the bounds over the ndgrid
[netMin,netMax,netPred] = conslearn.convex.computeICNNBoundsOverNDGrid(net,Vn,numOutputs);

% Refine lower bounds using fmincon
if options.RefineLowerBounds
    netMin = conslearn.convex.refineICNNLowerBoundsOverNDGrid(net,Vn,netMin);
end
end

function iValidateNetwork(net)
if ~isa(net.Layers(1),'nnet.cnn.layer.FeatureInputLayer')
    error("Network input layer must be a 'FeatureInputLayer'.")
end
numInputs = net.Layers(1).InputSize;
if numInputs > 3
    error("Number of network inputs must be less than or equal to 3.")
end
end

function iValidateHypercubicVertices(Vn)
for ii = 1:numel(Vn)
    if ~isnumeric(Vn{ii}) || any(isnan(Vn{ii}),"all") || any(isinf(Vn{ii}),"all")
        error("Each cell must be a numeric array of real, finite elements.")
    end
end
refSize = size(Vn{1});
for ii = 1:numel(Vn)
    if ~isequal(refSize,size(Vn{ii}))
        error("Size of cell element (" + ii + ") must match size of cell element one.")
    end
end
end

function Vn = iPreprocessVertices(Vn)
% If the number of inputs is 1, make sure the intervals are column vector
if isequal(numel(Vn),1)
    if isrow(Vn{1})
        Vn{1} = Vn{1}';
    end
end
end

function numOutputs = iComputeNumberChannelOutputs(net,Vn,numInputs)
X = [];
for ii = 1:numInputs
    X = [X Vn{ii}(:)]; %#ok<AGROW>
end
X = dlarray(X','CB');
% Sample a vertex
Z = predict(net,X(:,1));
cdim = finddim(Z,'C');
numOutputs = size(Z,cdim);
end

