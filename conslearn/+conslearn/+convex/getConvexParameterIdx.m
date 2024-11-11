function convexParameterIdx = getConvexParameterIdx(net)
% GETCONVEXPARAMETERIDX    Returns the indices in the learnable parameter
% table of the network that correspond to weights with convex constraints.
% The network *must* be created using the buildConstrainedNetwork or 
% buildConvexCNN function with a convex constraint type.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
end

convexParameterIdx = contains(net.Learnables.Layer,"_+_") & ...
    ( contains(net.Learnables.Parameter,"Weight") | contains(net.Learnables.Parameter,"Scale"));
end

