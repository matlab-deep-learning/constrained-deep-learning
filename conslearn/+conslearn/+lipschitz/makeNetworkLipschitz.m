function net = makeNetworkLipschitz(net,pnorm,lipschitzConstant)
% MAKENETWORKLIPSCHITZ    Constrain the weights of a Lipschitz network to
% ensure Lipschitz continuity of the outputs with respect to the network
% inputs. The network *must* be created using the buildConstrainedNetwork
% function with a Lipschitz constraint type.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    pnorm (1,1)
    lipschitzConstant (1,1) {mustBeNumeric,mustBeFinite,mustBePositive}
end

% Find the weights with constraints and compute the layer-wise constraints
[lipschitzParams,lipschitzIdx] = getLipschitzParameterIdx(net,lipschitzConstant);

% Apply the Lipschitz constraint
params = net.Learnables;
params(lipschitzIdx,:) = dlupdate(@(w,l)makeParametersLipschitz(w,l,pnorm),...
    params(lipschitzIdx,:),lipschitzParams(lipschitzIdx,:));
net.Learnables = params;
end

function [lipschitzParams,lipschitzIdx] = getLipschitzParameterIdx(net,lipschitzConstant)
[lipschitzParams,lipschitzIdx] = conslearn.lipschitz.getLipschitzParameterIdx(net,lipschitzConstant);
end

function renormalizedWeight = makeParametersLipschitz(weight,lipschitzConstant,pNorm)
renormalizedWeight = conslearn.lipschitz.makeParametersLipschitz(weight,lipschitzConstant,pNorm);
end