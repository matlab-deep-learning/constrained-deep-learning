function net = makeNetworkConvex(net)
% MAKENETWORKCONVEX    Constrain the weights of a convex network to ensure
% convexity of the outputs with respect to the network inputs. The network
% *must* be created using the buildConstrainedNetwork function with a convex
% constraint type.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
end

% Find the weights with constraint
convexParameterIdx = getConvexParameters(net);

% Apply the convex constraint
params = net.Learnables;
params(convexParameterIdx,:) = dlupdate(@(w)makeParametersConvex(w),...
    params(convexParameterIdx,:));
net.Learnables = params;
end

function convexParameterIdx = getConvexParameters(net)
convexParameterIdx = conslearn.convex.getConvexParameterIdx(net);
end

function renormalizedWeight = makeParametersConvex(weight)
renormalizedWeight = conslearn.convex.makeParametersConvex(weight);
end
