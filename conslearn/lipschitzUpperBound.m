function pLipschitzConstant = lipschitzUpperBound(net,p)
% LIPSCHITZUPPERBOUND    Lipschitz upper bound for neural network.
%
%   PLIP = LIPSCHITZUPPERBOUND(NET, P) returns a P-Lipschitz constant for
%   the dlnetwork object, NET, where NET *must* be a Lipschitz constrained
%   network constructed using the BUILDCONSTRAINEDNETWORK function with
%   Lipschitz constraint. P must be 1, 2, or Inf.

%   Copyright 2024 The MathWorks, Inc.

arguments
    net (1,1) dlnetwork
    p (1,1) {iValidatePNorm(p)}
end

% Initialize p-Lipschitz constant
pLipschitzConstant = 1;

% Extract the fully connected weight matrices index in the learnables table
% and their values
idx = contains(net.Learnables.Parameter,"Weights");
Ws = net.Learnables(idx,:).Value;

% Compute the lipschitz upper bound by taking the induced matrix p-norm on
% the weight matrices and multiplying these together
for ii = 1:numel(Ws)
    % p-norm of weight matrices
    pLipschitzConstant = pLipschitzConstant * conslearn.lipschitz.applyInducedPNorm(Ws{ii},p);
end
end

function iValidatePNorm(p)
if (~isequal(p,1) && ~isequal(p,2) && ~isequal(p,Inf)) && ~isempty(p)
error("Invalid 'PNorm' value. Value must be 1, 2, or Inf.")
end
end