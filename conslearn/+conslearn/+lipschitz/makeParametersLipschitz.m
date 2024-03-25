function Wp = makeParametersLipschitz(W,lambda,p)
% MAKEPARAMETERSLIPSCHITZ    Apply Lipschitz constraint to weights.

%   Copyright 2024 The MathWorks, Inc.

pnorm = applyInducedPNorm(W,p);
Wp = W./max(1,pnorm/lambda);
end

function pnorm = applyInducedPNorm(W,p)
pnorm = conslearn.lipschitz.applyInducedPNorm(W,p);
end