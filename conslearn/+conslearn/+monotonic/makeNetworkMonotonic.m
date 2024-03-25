function net = makeNetworkMonotonic(net,pnorm)
% MAKENETWORKMONOTONIC    Constrain the weights of a monotonic network to
% ensure montonicity of the outputs with respect to the network inputs. The
% network *must* be created using the buildConstrainedNetwork function with
% a monotonic constraint type.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    pnorm (1,1)
end

% Find the weights with constraint and apply monotonic constraint
lipschitzConstant = abs(net.Layers(end-1).ResidualScaling); 
[lipschitzParams,lipschitzIdx] = getLipschitzParameterIdx(net,lipschitzConstant);
params = net.Learnables;
params(lipschitzIdx,:) = dlupdate(@(w,l)makeParametersMonotonic(w,l,pnorm),...
    params(lipschitzIdx,:),lipschitzParams(lipschitzIdx,:));
net.Learnables = params;
end

function [lipschitzParams,lipschitzIdx] = getLipschitzParameterIdx(net,lipschitzConstant)
[lipschitzParams,lipschitzIdx] = conslearn.lipschitz.getLipschitzParameterIdx(net,lipschitzConstant);
end

function renormalizedWeight = makeParametersMonotonic(weight,lipschitzConstant,pnorm)
renormalizedWeight = conslearn.monotonic.makeParametersMonotonic(weight,lipschitzConstant,pnorm);
end