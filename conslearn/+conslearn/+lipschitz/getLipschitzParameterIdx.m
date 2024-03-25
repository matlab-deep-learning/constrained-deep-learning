function [lipschitzParams,lipschitzIdx] = getLipschitzParameterIdx(net,lipschitzConstant)
% GETLIPSCHITZPARAMETERIDX    Returns the indices in the learnable parameter
% table of the network that correspond to weights with Lipschitz
% constraints. Also returns the Lipschitz upper bounds in a learnables table
% format. The network *must* be created using the buildConstrainedNetwork
% function with a Lipschitz constraint type.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    lipschitzConstant (1,1) {mustBeNumeric,mustBeFinite,mustBePositive}
end
% The network architecture is a subset of pre-defined MLP architectures.
% Obtain the fully connected weight matrix indices.
lipschitzIdx = contains(net.Learnables.Parameter,"Weights");

% Initialize a parameter table for layer-wise Lipschitz upper bounds.
params = net.Learnables;
lipschitzParams = params;
lipschitzParams.Value = cell(size(params,1),1);

% Equally divide the Lipschitz constraint across learnable layers.
layerLipschitzConstant = dlarray(ones(size(params,1),1));
layerLipschitzConstant(lipschitzIdx) = layerLipschitzConstant(lipschitzIdx) *...
    (lipschitzConstant^(1/nnz(lipschitzIdx)));
lipschitzParams(lipschitzIdx,:).Value = num2cell(layerLipschitzConstant(lipschitzIdx));
end

