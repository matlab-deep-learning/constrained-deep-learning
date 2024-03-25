function Wp = makeParametersMonotonic(W,lambda,p)
% MAKEPARAMETERSMONOTONIC    Apply monotonic constraint to weights.

% - [1] Kitouni, Ouail, et al. Expressive Monotonic Neural Networks.
% arXiv:2307.07512, arXiv, 14 July 2023. arXiv.org,
% http://arxiv.org/abs/2307.07512.
%
%   Copyright 2024 The MathWorks, Inc.

if isequal(p,1)
    % 1-norm (column wise normalization [1] - equation 10)
    d = sum(abs(W),1)/lambda;
    d = 1./max(1, d);
    Wp = W.*d;
elseif isequal(p,2)
    % 2-norm
    pnorm = applyInducedPNorm(W,p);
    Wp = W./max(1,pnorm/lambda);
elseif isequal(p,Inf)
    % 1-norm (row wise normalization [1] - equation 10 adaption)
    d = sum(abs(W),2)/lambda;
    d = 1./max(1, d);
    Wp = d.*W;
end
end

function pnorm = applyInducedPNorm(W,p)
pnorm = conslearn.lipschitz.applyInducedPNorm(W,p);
end